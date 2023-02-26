import numpy as np
import torch
from torch.autograd import Variable

from . import scalers, nn


def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    if not torch.is_tensor(features):
        features = torch.tensor(features)
    bin_fractions = features[:, 2:4].cpu() % 1
    features = (features[:, :3].cpu() - torch.tensor([[0.0, 0.0, 162.5]])) / torch.tensor([[20.0, 60.0, 127.5]])
    return torch.cat((features, bin_fractions), dim=-1)


def disc_loss(d_real, d_fake):
    return torch.mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return torch.mean(d_real - d_fake)


class Model_v4(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self._f = preprocess_features
        if config['data_version'] == 'data_v4plus':
            self.full_feature_space = config.get('full_feature_space', False)
            self.include_pT_for_evaluation = config.get('include_pT_for_evaluation', False)
            if self.full_feature_space:
                self._f = preprocess_features_v4plus

        self.gp_lambda = config['gp_lambda']
        self.gpdata_lambda = config['gpdata_lambda']
        self.num_disc_updates = config['num_disc_updates']
        self.batch_size = config['batch_size']

        self.device = device

        self.stochastic_stepping = config['stochastic_stepping']
        self.dynamic_stepping = config.get('dynamic_stepping', False)
        if self.dynamic_stepping:
            assert not self.stochastic_stepping
            self.dynamic_stepping_threshold = config['dynamic_stepping_threshold']

        self.latent_dim = config['latent_dim']

        architecture_descr = config['architecture']

        self.generator = nn.FullModel(
            architecture_descr['generator'], custom_objects_code=config.get('custom_objects', None)
        ).to(self.device)
        self.discriminator = nn.FullModel(
            architecture_descr['discriminator'], custom_objects_code=config.get('custom_objects', None)
        ).to(self.device)

        self.disc_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=config['lr_disc'])
        self.gen_opt = torch.optim.RMSprop(self.generator.parameters(), lr=config['lr_gen'])

        self.step_counter = torch.tensor(0, dtype=torch.int)

        self.scaler = scalers.get_scaler(config['scaler'])
        self.pad_range = tuple(config['pad_range'])
        self.time_range = tuple(config['time_range'])
        self.data_version = config['data_version']

    def load_generator(self, model_checkpoint, opt_checkpoint):
        self._load_weights(model_checkpoint, opt_checkpoint, 'gen')

    def load_discriminator(self, model_checkpoint, opt_checkpoint):
        self._load_weights(model_checkpoint, opt_checkpoint, 'disc')

    def _load_weights(self, model_checkpoint, opt_checkpoint, gen_or_disc):
        network = self.generator
        optimizer = self.gen_opt
        if gen_or_disc == 'disc':
            network = self.discriminator
            optimizer = self.disc_opt
        elif gen_or_disc != 'gen':
            raise ValueError(gen_or_disc)

        print(f'Loading {gen_or_disc} weights from {str(model_checkpoint)}')
        network.load_state_dict(torch.load(model_checkpoint))
        optimizer.load_state_dict(torch.load(opt_checkpoint))

    def make_fake(self, features):
        size = len(features)
        latent_input = torch.normal(mean=0, std=1, size=(size, self.latent_dim), device=self.device)
        fake = self.generator(torch.cat((self._f(features).to(self.device), latent_input), dim=-1))
        return fake

    def gradient_penalty(self, features, real, fake):
        alpha = torch.rand(size=[len(real)] + [1] * (len(real.shape) - 1), device=self.device)
        fake = torch.reshape(fake, real.shape)
        interpolates = alpha * real + (1 - alpha) * fake

        interpolates = Variable(interpolates, requires_grad=True)
        processed_features = Variable(self._f(features).to(self.device), requires_grad=True)

        disc_interpolates = self.discriminator([processed_features, interpolates])
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True)[0]
    
        return torch.mean(torch.maximum(gradients.norm(2, dim=1) - 1, torch.tensor([0]).to(self.device)) ** 2)

    def gradient_penalty_on_data(self, features, real):
        d_real = self.discriminator([self._f(features), real])

        grads = torch.reshape(d_real.grad, (len(real), -1))
        return torch.mean(torch.sum(grads**2, dim=-1))

    def calculate_losses(self, feature_batch, target_batch):
        fake = self.make_fake(feature_batch)
        target_batch = torch.tensor(target_batch, device=self.device)
        feature_batch = self._f(feature_batch).to(self.device)
    
        d_real = self.discriminator([feature_batch, target_batch])
        d_fake = self.discriminator([feature_batch, fake])

        d_loss = disc_loss(d_real, d_fake)
        if self.training and self.gp_lambda > 0:
            penalty = self.gradient_penalty(feature_batch, target_batch, fake)
            d_loss = d_loss + penalty * self.gp_lambda
        if self.training and self.gpdata_lambda > 0:
            penalty = self.gradient_penalty_on_data(feature_batch, target_batch)
            d_loss = d_loss + penalty * self.gpdata_lambda

        g_loss = gen_loss(d_real, d_fake)
        return {'disc_loss': d_loss, 'gen_loss': g_loss}
        
    def disc_step(self, feature_batch, target_batch):
        self.disc_opt.zero_grad()
        
        losses = self.calculate_losses(feature_batch, target_batch)
        
        losses['disc_loss'].backward()
        self.disc_opt.step()

        return losses

    def gen_step(self, feature_batch, target_batch):
        self.gen_opt.zero_grad()
        
        losses = self.calculate_losses(feature_batch, target_batch)
        
        losses['gen_loss'].backward()        
        self.gen_opt.step()
        
        return losses
    
    def training_step(self, feature_batch, target_batch):
        if self.stochastic_stepping:
            if torch.randint(high=self.num_disc_updates + 1, size=(1,))[0] == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
            else:
                result = self.disc_step(feature_batch, target_batch)
        else:
            if self.step_counter == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
                self.step_counter.assign(0)
            else:
                result = self.disc_step(feature_batch, target_batch)
                if self.dynamic_stepping:
                    if result['disc_loss'] < self.dynamic_stepping_threshold:
                        self.step_counter.assign(self.num_disc_updates)
                else:
                    self.step_counter.assign_add(1)
        return result

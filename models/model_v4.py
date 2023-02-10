import h5py
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
    bin_fractions = features[:, 2:4] % 1
    device = bin_fractions.get_device()
    features = (features[:, :3] - torch.tensor([[0.0, 0.0, 162.5]], device=device)) / torch.tensor([[20.0, 60.0, 127.5]], device=device)
    return torch.cat((features, bin_fractions), dim=-1)


def preprocess_features_v4plus(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    #   padrow {23, 33}
    #   pT [0, 2.5]

    # print(features)
    bin_fractions = torch.tensor(features[:, 2:4] % 1)
    device = bin_fractions.get_device()
    features_1 = (features[:, :3] - torch.tensor([[0.0, 0.0, 162.5]], device=device)) / torch.tensor([[20.0, 60.0, 127.5]], device=device)
    features_2 = features[:, 4:5] >= 27
    features_3 = features[:, 5:6] / 2.5
    return torch.cat((features_1, features_2, features_3, bin_fractions), dim=-1)


def disc_loss(d_real, d_fake):
    return torch.mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return torch.mean(-d_fake)


class Model_v4:
    def __init__(self, config, device):
        self._f = preprocess_features
        if config['data_version'] == 'data_v4plus':
            self.full_feature_space = config.get('full_feature_space', False)
            self.include_pT_for_evaluation = config.get('include_pT_for_evaluation', False)
            if self.full_feature_space:
                self._f = preprocess_features_v4plus

        self.gp_lambda = config['gp_lambda']
        self.gpdata_lambda = config['gpdata_lambda']
        self.num_disc_updates = config['num_disc_updates']

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

    def load_generator(self, checkpoint):
        self._load_weights(checkpoint, 'gen')

    def load_discriminator(self, checkpoint):
        self._load_weights(checkpoint, 'disc')

    def _load_weights(self, checkpoint, gen_or_disc):
        if gen_or_disc == 'gen':
            network = self.generator
            step_fn = self.gen_step
        elif gen_or_disc == 'disc':
            network = self.discriminator
            step_fn = self.disc_step
        else:
            raise ValueError(gen_or_disc)

        model_file = h5py.File(checkpoint, 'r')
        if len(network.optimizer.weights) == 0 and 'optimizer_weights' in model_file:
            # perform single optimization step to init optimizer weights
            features_shape = self.discriminator.inputs[0].shape.as_list()
            targets_shape = self.discriminator.inputs[1].shape.as_list()
            features_shape[0], targets_shape[0] = 1, 1
            step_fn(torch.zeros(features_shape), torch.zeros(targets_shape))

        print(f'Loading {gen_or_disc} weights from {str(checkpoint)}')
        network.load_weights(str(checkpoint))

        if 'optimizer_weights' in model_file:
            print('Also recovering the optimizer state')
            opt_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(model_file) #????
            network.optimizer.set_weights(opt_weight_values)

    def make_fake(self, features):
        size = len(features)
        latent_input = torch.normal(mean=0, std=1, size=(size, self.latent_dim), device=self.device)
        return self.generator(torch.cat((self._f(features), latent_input), dim=-1))

    def gradient_penalty(self, features, real, fake):
        alpha = torch.rand(size=[len(real)] + [1] * (len(real.shape) - 1), device=self.device)
        fake = torch.reshape(fake, real.shape)
        interpolates = alpha * real + (1 - alpha) * fake
        
        inputs = [Variable(self._f(features), requires_grad=True), Variable(interpolates, requires_grad=True)]
        d_int = self.discriminator(inputs)
        grads = torch.reshape(interpolates.grad, (len(real), -1))
        return torch.mean(max(torch.norm(grads, dim=-1) - 1, 0) ** 2)

    def gradient_penalty_on_data(self, features, real):
        d_real = self.discriminator([self._f(features), real])

        grads = torch.reshape(d_real.grad, (len(real), -1))
        return torch.mean(torch.sum(grads**2, dim=-1))

    def calculate_losses(self, feature_batch, target_batch):
        feature_batch = torch.tensor(feature_batch, device=self.device)
        target_batch = torch.tensor(target_batch, device=self.device)
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([self._f(feature_batch), target_batch])
        d_fake = self.discriminator([self._f(feature_batch), fake])

        d_loss = disc_loss(d_real, d_fake)
        if self.gp_lambda > 0:
            d_loss = d_loss + self.gradient_penalty(feature_batch, target_batch, fake) * self.gp_lambda
        if self.gpdata_lambda > 0:
            d_loss = d_loss + self.gradient_penalty_on_data(feature_batch, target_batch) * self.gpdata_lambda

        g_loss = gen_loss(d_real, d_fake)
        return {'disk_loss': d_loss, 'gen_loss': g_loss}
        
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

import torch
import wandb

from metrics import make_images_for_model


class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period

    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Saving model on step {step} to {self.path}')
            gen_artifact = wandb.Artifact('generator', type='model')
            disc_artifact = wandb.Artifact('discriminator', type='model')
            gen_opt_artifact = wandb.Artifact('generator_optimizer', type='optimizer')
            disc_opt_artifact = wandb.Artifact('discriminator_optimizer', type='optimizer')

            torch.save(self.model.generator.state_dict(), str(self.path.joinpath("generator_checkpoint_{:05d}.pt".format(step))))
            torch.save(self.model.discriminator.state_dict(), str(self.path.joinpath("discriminator_checkpoint_{:05d}.pt".format(step))))
            torch.save(self.model.gen_opt.state_dict(), str(self.path.joinpath("gen_opt_{:05d}.pt".format(step))))
            torch.save(self.model.disc_opt.state_dict(), str(self.path.joinpath("disc_opt_{:05d}.pt".format(step))))
            
            gen_artifact.add_file(str(self.path.joinpath("generator_checkpoint_{:05d}.pt".format(step))))
            disc_artifact.add_file(str(self.path.joinpath("discriminator_checkpoint_{:05d}.pt".format(step))))
            gen_opt_artifact.add_file(str(self.path.joinpath("gen_opt_{:05d}.pt".format(step))))
            disc_opt_artifact.add_file(str(self.path.joinpath("disc_opt_{:05d}.pt".format(step))))
            
            wandb.log(gen_artifact)
            wandb.log(disc_artifact)
            wandb.log(gen_opt_artifact)
            wandb.log(disc_opt_artifact)


class EvaluateModelCallback:
    def __init__(self, model, path, validate_period, sample, batch_size):
        self.model = model
        self.path = path
        self.validate_period = validate_period
        self.sample = sample
        self.batch_size = batch_size

    def __call__(self, step):
        if step % self.validate_period == 0:
            print(f'Evaluating model on step {step}...')
            self.model.eval()
            with torch.no_grad():
                images, images1, img_amplitude, chi2 = make_images_for_model(self.model, sample=self.sample, batch_size=self.batch_size, calc_chi2=True)
                wandb.log({
                    'chi2': chi2,
                    'eval_epoch': step,
                })
                print(chi2)
                for k, img in images.items():
                    img_log = wandb.Image(img)
                    wandb.log({"is_masked": False, k: img_log, 'eval_epoch': step})
                for k, img in images1.items():
                    img_log = wandb.Image(img)
                    wandb.log({"is_masked": True, k: img_log, 'eval_epoch': step})
                img_log = wandb.Image(img_amplitude)
                wandb.log({"images with amplitude": img_log, 'eval_epoch': step})

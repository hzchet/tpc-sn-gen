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
            torch.save(self.model.generator.state_dict(), str(self.path.joinpath("generator_checkpoint_{:05d}.pt".format(step))))
            torch.save(self.model.discriminator.state_dict(), str(self.path.joinpath("discriminator_checkpoint_{:05d}.pt".format(step))))
            torch.save(self.model.gen_opt.state_dict(), str(self.path.joinpath("gen_opt_{:05d}.pt".format(step))))
            torch.save(self.model.disc_opt.state_dict(), str(self.path.joinpath("disc_opt_{:05d}.pt".format(step))))


class EvaluateModelCallback:
    def __init__(self, model, path, save_period, sample):
        self.model = model
        self.path = path
        self.save_period = save_period
        self.sample = sample

    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Evaluating model on step {step}...')
            self.model.eval()
            prediction_path = self.path / f"prediction_{step:05d}"
            with torch.no_grad():
                images, images1, img_amplitude, chi2, chi2_feature = make_images_for_model(self.model, sample=self.sample, calc_chi2=True)
                wandb.log({'chi2': chi2,
                           'chi2_Sigma1^2': chi2_feature,
                           'eval_epoch': step,
                           })
                print(chi2)
                print(chi2_feature)
                for k, img in images.items():
                    img_log = wandb.Image(img)
                    wandb.log({"images": img_log, 'eval_epoch': step})
                for k, img in images1.items():
                    img_log = wandb.Image(img)
                    wandb.log({"Masked images": img_log, 'eval_epoch': step})
                img_log = wandb.Image(img_amplitude)
                wandb.log({"images with amplitude": img_log, 'eval_epoch': step})

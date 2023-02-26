import torch
import torchvision.utils as vutils


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
            print(f'Evaluating model on step {step}')
            self.model.eval()
            prediction_path = self.path / f"prediction_{step:05d}"
            prediction_path.mkdir()
            with torch.no_grad():
                chi2 = evaluate_model(
                    self.model,
                    path=prediction_path,
                    sample=self.sample,
                    gen_sample_name=('generated.dat'),
                )
            wandb.log({
                f'eval_epoch': step,
                f'chi2': chi2,
            })

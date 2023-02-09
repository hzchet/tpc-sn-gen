#import tensorflow as tf
import torchvision.utils as vutils


class SaveModelCallback:
    def __init__(self, model, path, save_period):
        self.model = model
        self.path = path
        self.save_period = save_period


    def __call__(self, step):
        if step % self.save_period == 0:
            print(f'Saving model on step {step} to {self.path}')
            self.model.generator.save(str(self.path.joinpath("generator_{:05d}.h5".format(step))))
            self.model.discriminator.save(str(self.path.joinpath("discriminator_{:05d}.h5".format(step))))

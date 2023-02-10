import re
import torchvision.transforms as T
from torch.utils.data import Dataset


def epoch_from_name(name):
    (epoch,) = re.findall(r'\d+', name)
    return int(epoch)


def latest_epoch(model_path):
    gen_checkpoints = model_path.glob("generator_checkpoint_*.pt")
    disc_checkpoints = model_path.glob("discriminator_checkpoint*.pt")

    gen_epochs = [epoch_from_name(path.stem) for path in gen_checkpoints]
    disc_epochs = [epoch_from_name(path.stem) for path in disc_checkpoints]

    latest_gen_epoch = max(gen_epochs)
    latest_disc_epoch = max(disc_epochs)

    assert latest_gen_epoch == latest_disc_epoch, "Latest disc and gen epochs differ"

    return latest_gen_epoch


def load_weights(model, model_path, epoch=None):
    if epoch is None:
        epoch = latest_epoch(model_path)

    gen_checkpoint = model_path / f"generator_checkpoint_{epoch:05d}.pt"
    gen_opt_checkpoint = model_path / f"gen_opt_{epoch:05d}.pt"
    disc_checkpoint = model_path / f"discriminator_checkpoint_{epoch:05d}.pt"
    disc_opt_checkpoint = model_path / f"disc_opt_{epoch:05d}.pt"

    model.load_generator(gen_checkpoint, gen_opt_checkpoint)
    model.load_discriminator(disc_checkpoint, disc_opt_checkpoint)

    return epoch

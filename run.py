from pathlib import Path
import shutil
import argparse

import torch
import yaml
import wandb

from data import loader, preprocessing
from models.utils import load_weights
from models.training import train
from models.callbacks import SaveModelCallback, EvaluateModelCallback
from models.model_v4 import Model_v4


def make_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--logging_dir', type=str, default='logs')

    return parser


def print_args(args):
    print()
    print("----" * 10)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"    {k} : {v}")
    print("----" * 10)
    print()


def parse_args():
    args = make_parser().parse_args()
    print_args(args)
    return args


def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f)

    assert (config['feature_noise_power'] is None) == (
        config['feature_noise_decay'] is None
    ), 'Noise power and decay must be both provided'

    if 'lr_disc' not in config:
        config['lr_disc'] = config['lr']
    if 'lr_gen' not in config:
        config['lr_gen'] = config['lr']
    if 'lr_schedule_rate_disc' not in config:
        config['lr_schedule_rate_disc'] = config['lr_schedule_rate']
    if 'lr_schedule_rate_gen' not in config:
        config['lr_schedule_rate_gen'] = config['lr_schedule_rate']

    return config


def main():
    args = parse_args()
    
    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda:0')
    
    model_path = Path(args.logging_dir) / args.checkpoint_name

    config_path = str(model_path / 'config.yaml')
    continue_training = False

    if not args.config:
        assert model_path.exists(), "Couldn't find model directory"
        continue_training = True
    else:
        assert not model_path.exists(), "Model directory already exists"

        model_path.mkdir(parents=True)
        shutil.copy(args.config, config_path)

    args.config = config_path
    config = load_config(args.config)

    model = Model_v4(config, device).to(device)
    preprocessing._VERSION = model.data_version
    
    next_epoch = 0
    if continue_training:
        next_epoch = load_weights(model, model_path) + 1

    train_loader, X_test, Y_test = loader.get_loaders(
        scaler=model.scaler,
        batch_size=config['batch_size'],
        data_version=config['data_version'],
        pad_range=model.pad_range,
        time_range=model.time_range,
        strict=False
    )
    
    disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(model.disc_opt, gamma=config['lr_schedule_rate'])
    gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(model.gen_opt, gamma=config['lr_schedule_rate'])

    save_model = SaveModelCallback(model=model, path=model_path, save_period=config['save_every'])
    evaluate_model = EvaluateModelCallback(model=model, path=model_path, save_period=config['save_every'], sample=(X_test, Y_test))
    
    wandb.login(key='8e9008b623a334edf472f175d059c25c9aa66207')
    wandb.init(entity='hzchet', project='coursework', name=args.checkpoint_name)
    
    train(
        model,
        train_loader,
        Y_test,
        X_test,
        config['num_epochs'],
        config['batch_size'],
        gen_lr_scheduler,
        disc_lr_scheduler,
        first_epoch=next_epoch,
        callbacks=[save_model, evaluate_model],
    )


if __name__ == '__main__':
    main()

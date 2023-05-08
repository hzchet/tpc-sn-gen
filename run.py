from pathlib import Path
import shutil
import argparse

import torch
import yaml
from decouple import config as env_vars
import wandb

from data import loader
from models.utils import load_weights
from models.training import train
from models.callbacks import SaveModelCallback, EvaluateModelCallback
from models.model_v4 import Model_v4


def make_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--use_gpu', action='store_true', default=False)
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
        config = yaml.load(f, Loader=yaml.FullLoader)

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
    if args.prediction_only:
        assert model_path.exists(), "Couldn't find model directory"
        assert not args.config, "Config should be read from model path when doing prediction"
    else:
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

    next_epoch = 0
    if args.prediction_only or continue_training:
        next_epoch = load_weights(model, model_path) + 1

    train_loader, test_loader = loader.get_loaders(
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
    evaluate_model = EvaluateModelCallback(model=model, path=model_path, save_period=config['save_every'], sample=(X_test_raw, Y_test))
    
    wandb.login(key=env_vars('WANDB_API_KEY'))
    wandb.init(entity=env_vars('WANDB_ENTITY'), project=env_vars('WANDB_PROJECT'), name=args.checkpoint_name)
    
    train(
        model,
        train_loader,
        test_loader,
        config['num_epochs'],        
        gen_lr_scheduler,
        disc_lr_scheduler,
        first_epoch=next_epoch,
        callbacks=[save_model, evaluate_model],
    )


if __name__ == '__main__':
    main()

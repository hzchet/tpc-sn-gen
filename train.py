import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from models.model import Discriminator, Generator
from tqdm import tqdm
from data.preprocessing import read_csv_2d
from metrics import plotting
from models.utils import LoadData
from models.scalers import get_scaler
import torchvision.utils as vutils
import os
import wandb


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--checkpoint_name', type=str, default='baseline')
    parser.add_argument('--scaler', type=str, default='identity')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--disc_iters', type=int, defaul=5)
    parser.add_argument('--epochs', type=int, default=20)

    return parser


def parse_args():
    return make_parser().parse_args()


def train(generator, discriminator, loader, optimizer_g, optimizer_d, 
            scheduler_g, scheduler_d, num_epochs, disc_iters, latent_dim, device):
    generator.train()
    wandb.watch(generator)

    discriminator.train()
    wandb.watch(discriminator)

    loss_history = {'disc_losses': [], 'gen_losses': []}

    for epoch in num_epochs:
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            real_images = data.to(device)
            batch_size = real_images.size(0)
            noise = torch.normal(0, 1, size=(batch_size, latent_dim), device=device)
            
            # disc update
            discriminator.zero_grad()

            real_output = discriminator(batch)
            errD_real = torch.mean(real_output)

            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            errD_fake = torch.mean(fake_output)
            disc_loss = -errD_real + errD_fake
            disc_loss.backward()

            optimizer_d.step()

            # gen update
            if (i + 1) % disc_iters == 0:
                generator.zero_grad()

                fake_images = generator(noise)
                fake_output = discriminator(fake_images)
                gen_loss = -torch.mean(fake_output)
                gen_loss.backward()

                optimizer_g.step()

                loss_history['disc_losses'].append(disc_loss)
                loss_history['gen_losses'].append(gen_loss)

        disc_loss = torch.mean(torch.tensor(loss_history['disc_losses']))
        gen_loss = torch.mean(torch.tensor(loss_history['gen_losses']))
        
        print("epoch:", epoch)
        print("disc_loss:", disc_loss)
        print("gen_loss:", gen_loss)
        
        wandb.log({
            "Epoch": epoch,
            "gen Loss": gen_loss,
            "disc loss": disc_loss
        })


        if (epoch + 1) % 10 == 0:
            vutils.save_image(
                real_images,
                os.path.join("output", "real_samples.png"),
                normalize=True
            )
            fake = generator(torch.normal(0, 1, size=(batch_size, latent_dim), device=device))
            vutils.save_image(
                fake.detach(),
                os.path.join("output", f"fake_samples_{epoch + 1}.png"),
                normalize=True
            )
        
        scheduler_d.step()
        scheduler_g.step()



if __name__ == '__main__':
    wandb.login()
    wandb.init(project='coursework', entity='hzchet')

    args = parse_args()
    device = torch.device('cuda:0')
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    scheduler_g = torch.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
    scheduler_d = torch.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)
    
    scaler = get_scaler(scaler_type=args.scaler)
    data, _ = read_csv_2d(filename='data/data_v2/csv/digits.csv', strict=False)
    dataset = LoadData(np.float32(scaler.scale(data)), T.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    
    train(
        generator, 
        discriminator, 
        loader, 
        optimizer_g, 
        optimizer_d, 
        scheduler_g, 
        scheduler_d,
        args.epochs,
        args.disc_iters,
        args.latent_dim,
        device
    )

    wandb.finish()

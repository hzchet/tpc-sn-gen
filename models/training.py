import numpy as np
from tqdm import trange
import torch
import wandb


def train(
    model,
    train_loader,
    test_loader,
    num_epochs,
    gen_lr_scheduler=None,
    disc_lr_scheduler=None,
    first_epoch=0,
    callbacks=None,
):
    
    train_gen_losses = []
    train_disc_losses = []
    val_gen_losses = []
    val_disc_losses = []

    for i_epoch in range(first_epoch, num_epochs):
        print("Working on epoch #{}".format(i_epoch), flush=True)

        model.train()
        
        losses_train = {}

        for data, features in trange(train_loader):
            losses_train_batch = model.training_step(data, features)
            for k, l in losses_train_batch.items():
                losses_train[k] = losses_train.get(k, 0) + l.item() * len(data)

        if disc_lr_scheduler is not None:
            disc_lr_scheduler.step()

        if gen_lr_scheduler is not None:
            gen_lr_scheduler.step()

        losses_train = {k: l / len(train_loader.dataset) for k, l in losses_train.items()}
        train_gen_losses.append(losses_train['gen_loss'])
        train_disc_losses.append(losses_train['disc_loss'])
        wandb.log({
            'train_gen_loss': losses_train['gen_loss'],
            'train_disc_loss': losses_train['disc_loss'],
            'epoch': i_epoch, 
        })
        
        model.eval()
        losses_val = {}
        for data, features in trange(test_loader):
            with torch.no_grad():
                losses_val_batch = {k: l.cpu().detach().numpy() for k, l in model.calculate_losses(data, features).items()}
                for k, l in losses_val_batch.items():
                    losses_val[k] = losses_val.get(k, 0) + l * len(data)

        losses_val = {k: l / len(train_loader.dataset) for k, l in losses_val.items()}
        val_gen_losses.append(losses_val['gen_loss'])
        val_disc_losses.append(losses_val['disc_loss'])
        
        wandb.log({
            'val_gen_loss': losses_val['gen_loss'],
            'val_disc_loss': losses_val['disc_loss'],
            'epoch': i_epoch,
        })

        if callbacks is not None:
            for f in callbacks:
                f(i_epoch)

        print("", flush=True)
        print("Train losses:", losses_train)
        print("Val losses:", losses_val)

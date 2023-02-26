import numpy as np
from tqdm import trange
import torch
import wandb


def train(
    model,
    data_train,
    data_val,
    num_epochs,
    batch_size,
    gen_lr_scheduler=None,
    disc_lr_scheduler=None,
    features_train=None,
    features_val=None,
    features_noise=None,
    first_epoch=0,
    callbacks=None,
):
    if not ((features_train is None) or (features_val is None)):
        assert features_train is not None, 'train: features should be provided for both train and val'
        assert features_val is not None, 'train: features should be provided for both train and val'

    train_gen_losses = []
    train_disc_losses = []
    val_gen_losses = []
    val_disc_losses = []

    for i_epoch in range(first_epoch, num_epochs):
        print("Working on epoch #{}".format(i_epoch), flush=True)

        model.train()
        
        shuffle_ids = np.random.permutation(len(data_train))
        losses_train = {}

        noise_power = None
        if features_noise is not None:
            noise_power = features_noise(i_epoch)

        for i_sample in trange(0, len(data_train), batch_size):
            batch = data_train[shuffle_ids][i_sample : i_sample + batch_size]
            if features_train is not None:
                feature_batch = features_train[shuffle_ids][i_sample : i_sample + batch_size]
                if noise_power is not None:
                    feature_batch = (
                        feature_batch
                        + np.random.normal(size=feature_batch.shape).astype(feature_batch.dtype) * noise_power
                    )

            if features_train is None:
                losses_train_batch = model.training_step(batch)
            else:
                losses_train_batch = model.training_step(feature_batch, batch)
            for k, l in losses_train_batch.items():
                losses_train[k] = losses_train.get(k, 0) + l.item() * len(batch)

        if disc_lr_scheduler is not None:
            disc_lr_scheduler.step()

        if gen_lr_scheduler is not None:
            gen_lr_scheduler.step()

        losses_train = {k: l / len(data_train) for k, l in losses_train.items()}
        train_gen_losses.append(losses_train['gen_loss'])
        train_disc_losses.append(losses_train['disc_loss'])
        wandb.log({
            'train_gen_loss': losses_train['gen_loss'],
            'train_disc_loss': losses_train['disc_loss'],
            'epoch': i_epoch, 
        })
        
        model.eval()
        losses_val = {}
        for i_sample in trange(0, len(data_val), batch_size):
            batch = data_val[i_sample : i_sample + batch_size]
            with torch.no_grad():
                if features_train is None:
                    losses_val_batch = {k: l.cpu().detach().numpy() for k, l in model.calculate_losses(batch).items()}
                else:
                    feature_batch = features_val[i_sample : i_sample + batch_size]
                    losses_val_batch = {k: l.cpu().detach().numpy() for k, l in model.calculate_losses(feature_batch, batch).items()}
            for k, l in losses_val_batch.items():
                losses_val[k] = losses_val.get(k, 0) + l * len(batch)
        losses_val = {k: l / len(data_val) for k, l in losses_val.items()}
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
        

def average(models):
    parameters = [model.parameters() for model in models]
    assert len(np.unique([len(par) for par in parameters])) == 1, 'average: different models provided'

    result = torch.clone(models[0])
    for params in zip(result.parameters(), *parameters):
        params[0].assign(np.mean(params[1:], axis=0))

    return result

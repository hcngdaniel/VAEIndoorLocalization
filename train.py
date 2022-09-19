#!/usr/bin/env python3
import os
from types import SimpleNamespace
import yaml
import itertools

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset
from models.model1 import Model, LossFunc


# load config
def to_simple_namespace(d):
    for key, value in filter(lambda x: isinstance(x[1], dict), d.items()):
        d[key] = to_simple_namespace(value)
    return SimpleNamespace(**d)


with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    config = to_simple_namespace(config)

# load device
device = config.general.device
if not torch.cuda.is_available():
    device = 'cpu'
device = torch.device(device)
print(f'using device {device}')

# load dataset
dataset = Dataset(
    dataset_name=config.dataset.dataset_name,
)

# initialize dataloader
dataloader = DataLoader(
    dataset,
    batch_size=config.dataloader.batch_size,
    shuffle=config.dataloader.shuffle,
    num_workers=config.dataloader.num_workers,
    drop_last=config.dataloader.drop_last,
)

# load model
model = Model()
if os.path.exists(os.path.join('saves', config.model.save_name)):
    model.load_state_dict(torch.load(os.path.join('saves', config.model.save_name)))
model.to(device)
model.train()
encoder = model.encoder
decoder = model.decoder
sampler = model.sampler
transformer = model.transformer

# load loss function
loss_fn = LossFunc()

# load optimizer
vae_optimizer = optim.Adagrad(
    itertools.chain(encoder.parameters(), decoder.parameters()),
    lr=config.optimizer.lr,
)
transformer_optimizer = optim.Adagrad(
    transformer.parameters(),
    lr=config.optimizer.lr,
)

# initialize tensorboard
summary_writer = SummaryWriter(
    log_dir=os.path.join('logs', config.logs.log_dir),
    flush_secs=config.logs.flush_secs,
)


# define train
def train_vae(epoch):
    encoder.train()
    decoder.train()
    transformer.eval()

    running_reconstruction_loss = 0
    running_kld_loss = 0
    running_vae_loss = 0

    for batch, (image, (target_norm, target_theta, target_alpha)) in enumerate(dataloader, 1):
        # to device
        image = image.to(device)

        # encode
        mu, logvar = encoder(image)
        # sample
        z = sampler(mu, logvar)
        # decode
        reconstructed = decoder(z)

        # calculate loss
        reconstruction_loss = loss_fn.reconstruction_loss(
            reconstructed, image,
        )
        kld_loss = loss_fn.kld_loss(
            mu, logvar,
        )
        loss = reconstruction_loss + kld_loss

        # optimize
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()

        # calculate running loss
        running_reconstruction_loss = (running_reconstruction_loss * (batch - 1) +
                                       reconstruction_loss.detach().cpu().numpy()) / batch
        running_kld_loss = (running_kld_loss * (batch - 1) +
                            kld_loss.detach().cpu().numpy()) / batch
        running_vae_loss = (running_vae_loss * (batch - 1) +
                            loss.detach().cpu().numpy()) / batch

        print(f'vae loss: {running_vae_loss:.6f}')

    # save loss
    summary_writer.add_scalar(
        "vae_loss",
        running_vae_loss,
        epoch,
    )
    summary_writer.add_scalar(
        "reconstruction_loss",
        running_reconstruction_loss,
        epoch,
    )
    summary_writer.add_scalar(
        "kld_loss",
        running_kld_loss,
        epoch,
    )


def train_transformer(epoch):
    encoder.eval()
    decoder.eval()
    transformer.train()
    running_transformer_loss = 0
    for batch, (image, (target_norm, target_theta, target_alpha)) in enumerate(dataloader, 1):
        # to device
        image = image.to(device)
        target_norm = target_norm.to(device)
        target_theta = target_theta.to(device)
        target_alpha = target_alpha.to(device)

        # encode
        mu, logvar = encoder(image)
        # transform
        norm, theta, alpha = transformer(mu)

        # calculate loss
        loss = loss_fn.transformer_loss(
            norm, target_norm,
            theta, target_theta,
            alpha, target_alpha,
        )

        # optimize
        transformer_optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()

        running_transformer_loss = (running_transformer_loss * (batch - 1) +
                                    loss.detach().cpu().numpy()) / batch

        print(f"transformer loss: {running_transformer_loss:.6f}")

    summary_writer.add_scalar(
        "transformer_loss",
        running_transformer_loss,
        epoch,
    )


def save():
    torch.save(model.state_dict(), os.path.join('saves', config.model.save_name))


for epoch in range(config.general.start_epoch, config.general.end_epoch):
    try:
        train_vae(epoch)
        train_transformer(epoch)
        if epoch % config.general.n_epoch_per_save == 0:
            save()
            print('saved')
    except KeyboardInterrupt:
        with open('train_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config["general"]["start_epoch"], \
        config["general"]["end_epoch"] = \
            epoch, \
            epoch + config["general"]["end_epoch"] - config["general"]["start_epoch"]
        with open('train_config.yaml', 'w') as f:
            yaml.safe_dump(config, f)
        exit()

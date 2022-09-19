#!/usr/bin/env python3
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 10, 9, 1, 4),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(10, 15, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(15, 20, 5, 1, 2, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(20),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(20, 15, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(15),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(15, 10, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(10, 5, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(5, 5, 5, 1, 2),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((10, 15)),
            nn.Flatten(),
        )
        self.mu_fcl = nn.Sequential(
            nn.Linear(750, 128),
            nn.LeakyReLU(0.2, True),
        )
        self.logvar_fcl = nn.Sequential(
            nn.Linear(750, 128),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.mu_fcl(backbone_out), self.logvar_fcl(backbone_out)


class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    @classmethod
    def forward(cls, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 750),
            nn.LeakyReLU(0.2, True),
            nn.Unflatten(dim=1, unflattened_size=(5, 10, 15)),
            nn.Upsample((15, 20)),
            nn.ConvTranspose2d(5, 5, 5, 1, 2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(5, 10, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample((30, 40)),
            nn.ConvTranspose2d(10, 15, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(15),
            nn.Upsample((60, 80)),
            nn.ConvTranspose2d(15, 20, 3, 1, 1, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(20),
            nn.Upsample((120, 160)),
            nn.ConvTranspose2d(20, 15, 5, 1, 2, bias=False),
            nn.SiLU(),
            nn.BatchNorm2d(15),
            nn.Upsample((240, 320)),
            nn.ConvTranspose2d(15, 10, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample((480, 640)),
            nn.ConvTranspose2d(10, 3, 9, 1, 4),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.main_fcl = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, True),
        )
        self.norm_fcl = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.theta_fcl = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.alpha_fcl = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, mu):
        main_fcl_out = self.main_fcl(mu)
        return self.norm_fcl(main_fcl_out), self.theta_fcl(main_fcl_out), self.alpha_fcl(main_fcl_out)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.sampler = Sampler()
        self.decoder = Decoder()
        self.transformer = Transformer()


class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='mean')

    @classmethod
    def kld_loss(cls, mu, logvar):
        return torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))

    @classmethod
    def transformer_loss(cls, norm, target_norm, theta, target_theta, alpha, target_alpha):
        return torch.mean((norm - target_norm) ** 2) + \
               torch.mean((theta - target_theta) ** 2) + \
               torch.mean((alpha - target_alpha) ** 2)

    def forward(self, x, target_x, mu, logvar, norm, target_norm, theta, target_theta, alpha, target_alpha):
        return self.reconstruction_loss(x, target_x), self.kld_loss(mu, logvar), \
               self.transformer_loss(norm, target_norm, theta, target_theta, alpha, target_alpha)

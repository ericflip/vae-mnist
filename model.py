import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # (1, 28, 28) -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # (16, 14, 14) -> (32, 7, 7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # (32, 7, 7) -> (64, 4, 4)
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x: torch.Tensor):
        # apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # flatten
        x = x.flatten(start_dim=1, end_dim=-1)

        # get mu and log var
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # sample
        eps = torch.normal(mean=0, std=1, size=logvar.shape).to(x.device)
        z = mu + torch.exp(logvar / 2) * eps

        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.ztransform = nn.Linear(latent_dim, 1024)
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),  # (64, 4, 4) -> (32, 7, 7)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 2, 1),  # (32, 7, 7) -> (16, 14, 14)
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),  # (16, 14, 14) -> (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        # transform z to 2d
        z = self.ztransform(z)
        z = z.view(-1, 64, 4, 4)

        # transpose convolutions
        z = self.tconv1(z)
        z = self.tconv2(z)
        z = self.tconv3(z)

        return z


class MNISTVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def encode():
        pass

    def decode():
        pass

    def sample():
        pass

    def generate():
        pass

    def forward(self, x: torch.Tensor):
        pass

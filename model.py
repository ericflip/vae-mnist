import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims):
        super().__init__()

    def forward():
        pass


class Decoder(nn.Module):
    pass


class VAE(nn.Module):
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

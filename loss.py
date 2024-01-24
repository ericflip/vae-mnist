import torch
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self, kld_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.kld_weight = kld_weight

    def __call__(self, x, x_hat, mu, logvar):
        l1 = self.mse(x, x_hat)
        l2 = self.kld_weight * torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=-1)
        )

        loss = l1 + l2

        return loss

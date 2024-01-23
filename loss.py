import torch
import torch.nn as nn


class VAELoss(nn.Module):
    mse = nn.MSELoss()

    def __call__(self, x, x_hat, mu, logvar):
        l1 = self.mse(x, x_hat)
        l2 = 0.01 * torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=-1)
        )

        loss = l1 + l2

        return loss

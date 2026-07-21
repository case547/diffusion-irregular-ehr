import torch
import torch.nn as nn
from pyro.contrib.cevae import BernoulliNet, DiagNormalNet
from torch.distributions import Bernoulli, Normal


class XDecoder(nn.Module):
    """p_ψ(x | z) -- diagonal Gaussian. Training only; not used at inference."""

    def __init__(self, latent_dim: int, feature_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        sizes = [latent_dim] + [hidden_dim] * (num_layers - 1) + [feature_dim]
        self.net = DiagNormalNet(sizes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loc, scale) each (B, feature_dim)."""
        return self.net(z)

    def log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Joint log-prob under diagonal Gaussian, summed over feature dim. Shape (B,)."""
        loc, scale = self.forward(z)
        return Normal(loc, scale).log_prob(x).sum(dim=-1)


class ADecoder(nn.Module):
    """p_ψ(a | z) -- Bernoulli. Training only; not used at inference."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        sizes = [latent_dim] + [hidden_dim] * (num_layers - 1)
        self.net = BernoulliNet(sizes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits (B,)."""
        (logits,) = self.net(z)
        return logits

    def log_prob(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Bernoulli log-prob (B,)."""
        return Bernoulli(logits=self.forward(z)).log_prob(a)

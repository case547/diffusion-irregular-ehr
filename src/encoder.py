import torch
import torch.nn as nn
from pyro.contrib.cevae import DiagNormalNet, FullyConnected


class ZEncoder(nn.Module):
    """r_φ(z | x, a, y_fac) -- TARnet-split diagonal Gaussian encoder.

    Shared trunk g1 processes cat([x, y_fac]).
    Treatment heads g2 (a=0) and g3 (a=1) each output (mu, sigma) via DiagNormalNet.
    torch.where selects the active branch; gradients flow only through the matching head.
    """

    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        trunk_sizes = [feature_dim + 1] + [hidden_dim] * num_layers
        self.trunk = FullyConnected(trunk_sizes, final_activation=nn.ELU())
        self.head0 = DiagNormalNet([hidden_dim, latent_dim])  # a=0
        self.head1 = DiagNormalNet([hidden_dim, latent_dim])  # a=1

    def forward(
        self, x: torch.Tensor, a: torch.Tensor, y_fac: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:     (B, feature_dim)
        a:     (B,) float binary {0, 1}
        y_fac: (B,) observed factual outcome
        Returns (mu, sigma) each (B, latent_dim).
        """
        hidden = self.trunk(torch.cat([x, y_fac.unsqueeze(-1)], dim=-1))
        mu0, sigma0 = self.head0(hidden)
        mu1, sigma1 = self.head1(hidden)
        sel = a.bool().unsqueeze(-1)  # (B, 1) for broadcasting
        return torch.where(sel, mu1, mu0), torch.where(sel, sigma1, sigma0)

    def rsample(
        self, x: torch.Tensor, a: torch.Tensor, y_fac: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterised sample. Returns (z, mu, sigma)."""
        mu, sigma = self.forward(x, a, y_fac)
        z = mu + sigma * torch.randn_like(mu)
        return z, mu, sigma

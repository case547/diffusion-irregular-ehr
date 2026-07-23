import torch
import torch.nn as nn
from pyro.contrib.cevae import BernoulliNet, FullyConnected, NormalNet
from torch.distributions import Bernoulli, Normal


class AuxTreatment(nn.Module):
    """r_φ(a | x) -- used to marginalise over treatment at inference."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        sizes = [feature_dim] + [hidden_dim] * (num_layers - 1)
        self.net = BernoulliNet(sizes)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        (logits,) = self.net(x)
        return logits

    def log_prob(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(B,)"""
        return Bernoulli(logits=self._logits(x)).log_prob(a)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Binary float sample (B,)."""
        return Bernoulli(logits=self._logits(x)).sample()


class AuxOutcome(nn.Module):
    """r_φ(y | x, a) -- TARnet-split; used to marginalise over outcome at inference.

    Shared trunk takes x; treatment-specific heads select via torch.where.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        trunk_sizes = [feature_dim] + [hidden_dim] * (num_layers - 1)
        self.trunk = FullyConnected(trunk_sizes, final_activation=nn.ELU())
        self.head0 = NormalNet([hidden_dim])  # a=0
        self.head1 = NormalNet([hidden_dim])  # a=1

    def _params(self, x: torch.Tensor, a: torch.Tensor):
        hidden = self.trunk(x)
        loc0, scale0 = self.head0(hidden)
        loc1, scale1 = self.head1(hidden)
        sel = a.bool()
        return torch.where(sel, loc1, loc0), torch.where(sel, scale1, scale0)

    def log_prob(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """(B,)"""
        loc, scale = self._params(x, a)
        return Normal(loc, scale).log_prob(y)

    def sample(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(B,)"""
        loc, scale = self._params(x, a)
        return Normal(loc, scale).sample()

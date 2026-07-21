import numpy as np
import torch

from src.propensity import PropensityNet

B, F = 64, 25


def _data(n=B, f=F):
    x = torch.randn(n, f)
    a = torch.randint(0, 2, (n,)).long()
    return x, a


def test_forward_output_shape():
    net = PropensityNet(n_unit_in=F, n_units_out_prop=32, n_layers_out_prop=0)
    x, _ = _data()
    assert net(x).shape == (B, 2)


def test_forward_probs_sum_to_one():
    net = PropensityNet(n_unit_in=F, n_units_out_prop=32, n_layers_out_prop=0)
    x, _ = _data()
    assert torch.allclose(net(x).sum(dim=-1), torch.ones(B), atol=1e-5)


def test_importance_weights_shape():
    net = PropensityNet(n_unit_in=F, n_units_out_prop=32, n_layers_out_prop=0)
    x, a = _data()
    w = net.get_importance_weights(x, a.float())
    assert w.shape == (B,)
    assert (w > 0).all()


def test_fit_reduces_loss():
    torch.manual_seed(0)
    np.random.seed(0)
    net = PropensityNet(
        n_unit_in=F,
        n_units_out_prop=32,
        n_layers_out_prop=0,
        n_iter=100,
        batch_size=64,
        val_split_prop=0.0,
    )
    x, a = _data(n=128)
    loss_before = net.loss(net(x), a).item()
    net.fit(x, a)
    loss_after = net.loss(net(x), a).item()
    assert loss_after < loss_before

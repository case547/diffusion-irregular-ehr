import torch

from src.encoder import ZEncoder

B, F, D = 8, 25, 20


def _batch():
    return torch.randn(B, F), torch.randint(0, 2, (B,)).float(), torch.randn(B)


def test_output_shapes():
    enc = ZEncoder(feature_dim=F, latent_dim=D, hidden_dim=64, num_layers=2)
    mu, sigma = enc(*_batch())
    assert mu.shape == (B, D)
    assert sigma.shape == (B, D)


def test_sigma_positive():
    enc = ZEncoder(feature_dim=F, latent_dim=D, hidden_dim=64, num_layers=2)
    _, sigma = enc(*_batch())
    assert (sigma > 0).all()


def test_rsample_shape():
    enc = ZEncoder(feature_dim=F, latent_dim=D, hidden_dim=64, num_layers=2)
    z, mu, sigma = enc.rsample(*_batch())
    assert z.shape == (B, D)
    assert mu.shape == (B, D)


def test_rsample_gradient_flows():
    enc = ZEncoder(feature_dim=F, latent_dim=D, hidden_dim=64, num_layers=2)
    z, _, _ = enc.rsample(*_batch())
    z.sum().backward()
    # Gradient reaches the trunk's first Linear weight via reparameterisation
    assert enc.trunk[0].weight.grad is not None


def test_tarnet_heads_differ():
    enc = ZEncoder(feature_dim=F, latent_dim=D, hidden_dim=64, num_layers=2)
    x, _, y = _batch()
    mu0, _ = enc(x, torch.zeros(B), y)
    mu1, _ = enc(x, torch.ones(B), y)
    assert not torch.allclose(mu0, mu1)

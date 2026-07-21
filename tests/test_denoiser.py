import torch
from src.denoiser import MLP, Denoiser

B, D, L = 8, 20, 100


def _inputs():
    return (
        torch.randn(B, 2),
        torch.randint(0, L, (B,)),
        torch.randn(B, D),
        torch.randint(0, 2, (B,)).float(),
    )


def test_mlp_output_shape():
    mlp = MLP(in_dim=16, hidden_dim=32, out_dim=16)
    x = torch.randn(B, 16)
    out = mlp(x)
    assert out.shape == (B, 16)
    assert torch.isfinite(out).all()


def test_mlp_gradient_flows():
    mlp = MLP(in_dim=16, hidden_dim=32, out_dim=16)
    mlp(torch.randn(B, 16)).sum().backward()
    assert mlp.net[0].weight.grad is not None


def test_output_shape():
    m = Denoiser(latent_dim=D, block_dim=32, hidden_dim=32, embedding_dim=64, num_blocks=2, num_steps=L)
    eps = m(*_inputs())
    assert eps.shape == (B, 2)
    assert torch.isfinite(eps).all()


def test_gradient_flows():
    m = Denoiser(latent_dim=D, block_dim=32, hidden_dim=32, embedding_dim=64, num_blocks=2, num_steps=L)
    m(*_inputs()).sum().backward()
    assert m.cond_proj.weight.grad is not None

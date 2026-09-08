import numpy as np
import torch

from src.config import DiffusionConfig, VAEConfig
from src.confounding_analysis import (
    boot_curve,
    cross_evaluate,
    divergence_ratio,
    pointwise_ci,
)
from src.model import DiffPO

VAE_CFG = VAEConfig(
    feature_dim=5,
    latent_dim=4,
    hidden_dim=16,
    encoder_num_layers=2,
    decoder_num_layers=1,
    aux_num_layers=1,
    a_decoder_hidden_dim=5,
)
DIFF_CFG = DiffusionConfig(
    num_steps=10,
    beta_start=0.0001,
    beta_end=0.02,
    schedule="quad",
    embedding_dim=16,
    block_dim=16,
    hidden_dim=32,
    num_blocks=2,
)
L, N, F = DIFF_CFG.num_steps, 4, 5


def _model_and_traj():
    model = DiffPO(VAE_CFG, DIFF_CFG)
    x = torch.randn(N, F)
    a = torch.randint(0, 2, (N,)).float()
    _, y_traj, eps_traj = model.sample_ddim(x, a, log_trajectory=True)
    return model, x, a, y_traj, eps_traj


def test_cross_evaluate_shape():
    model, x, a, y_traj, _ = _model_and_traj()
    out = cross_evaluate(model, y_traj, model.encode_cond(x, a), a)
    assert out.shape == (L, N, 2)


def test_cross_evaluate_deterministic():
    model, x, a, y_traj, _ = _model_and_traj()
    cond = model.encode_cond(x, a)
    assert torch.equal(
        cross_evaluate(model, y_traj, cond, a), cross_evaluate(model, y_traj, cond, a)
    )


def test_cross_evaluate_reproduces_own_eps_on_own_trajectory():
    """A model cross-evaluated on its own logged trajectory with its own conditioning
    must reproduce its own logged predicted noise (up to float error)."""
    model, x, a, y_traj, eps_traj = _model_and_traj()
    out = cross_evaluate(model, y_traj, model.encode_cond(x, a), a)
    assert torch.allclose(out, eps_traj, atol=1e-5)


def test_divergence_ratio_matches_manual():
    div = torch.rand(L, N) + 0.5
    eps = torch.rand(L, N) + 0.5
    mask = torch.tensor([True, False, True, False])
    expected = (div[:, mask].mean() / eps[:, mask].mean()).item()
    assert divergence_ratio(div, eps, mask) == expected


def test_boot_curve_and_pointwise_ci_shapes():
    values = np.random.default_rng(0).normal(size=(L, 7))
    boot = boot_curve(values, n_boot=50, rng=np.random.default_rng(1))
    assert boot.shape == (L, 50)
    lo, hi = pointwise_ci(boot)
    assert lo.shape == (L,) and hi.shape == (L,)
    assert np.all(lo <= hi)

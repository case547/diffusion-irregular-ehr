import pytest
import torch

from src.config import DiffusionConfig, ModelConfig
from src.model import DiffPO
from src.propensity import PropensityNet

MODEL_CFG = ModelConfig(feature_dim=5, latent_dim=4, hidden_dim=16, num_layers=2)
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
B, F = 4, 5


def _batch():
    return torch.randn(B, F), torch.randint(0, 2, (B,)).float(), torch.randn(B), torch.randn(B)


def test_loss_keys_and_finite():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    assert set(comps.keys()) == {"diffusion_loss"}
    assert comps["diffusion_loss"].shape == ()
    assert torch.isfinite(comps["diffusion_loss"])


def test_loss_with_propnet():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    propnet = PropensityNet(
        n_unit_in=F,
        n_units_out_prop=16,
        n_layers_out_prop=0,
        batch_norm=False,
    )
    x, a, y, y_cf = _batch()
    comps = model.compute_loss(x, a, y, y_cf, propnet=propnet)
    assert torch.isfinite(comps["diffusion_loss"])


def test_cf_anchor_inert_by_default():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x, a, y_fac, y_cf = _batch()
    torch.manual_seed(0)
    comps_anchored = model.compute_loss(x, a, y_fac, y_cf, pop_means=(3.0, -2.0))
    torch.manual_seed(0)
    comps_plain = model.compute_loss(x, a, y_fac, y_cf)
    assert comps_anchored["diffusion_loss"].item() == pytest.approx(
        comps_plain["diffusion_loss"].item()
    )


def test_cf_anchor_uses_population_mean_not_y_cf():
    """Spy on the real _noise_targets call inside compute_loss -- directly verifies what
    cf_target compute_loss actually constructed, matching tests/test_model.py's convention
    for the same check on HybridModel."""
    cfg = DiffusionConfig(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
        cf_anchor_weight=0.1,
    )
    model = DiffPO(MODEL_CFG, cfg)
    x = torch.randn(B, F)
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y_fac = torch.randn(B)
    y_cf = torch.full((B,), 999.0)  # deliberately extreme -- must NOT reach _noise_targets
    pm0, pm1 = 3.0, -2.0

    captured = {}
    original = model._noise_targets

    def spy(batch_size, device, a_arg, y_fac_arg, y_cf_arg):
        captured["y_cf_arg"] = y_cf_arg
        return original(batch_size, device, a_arg, y_fac_arg, y_cf_arg)

    model._noise_targets = spy
    model.compute_loss(x, a, y_fac, y_cf, pop_means=(pm0, pm1))

    expected_cf_target = torch.where(a == 1, torch.full_like(a, pm0), torch.full_like(a, pm1))
    assert torch.allclose(captured["y_cf_arg"], expected_cf_target)
    assert not torch.allclose(captured["y_cf_arg"], y_cf)


def test_cf_anchor_finite_loss():
    cfg = DiffusionConfig(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
        cf_anchor_weight=0.1,
    )
    model = DiffPO(MODEL_CFG, cfg)
    x, a, y_fac, y_cf = _batch()
    comps = model.compute_loss(x, a, y_fac, y_cf, pop_means=(3.0, -2.0))
    assert torch.isfinite(comps["diffusion_loss"])


def test_cf_anchor_softens_gradient_mask_in_compute_loss():
    """Spy on the real calculate_diffusion_loss call inside compute_loss -- verifies the
    ACTUAL mask tensor reaching it, not just a standalone recomputation of the formula
    (mirrors tests/test_model.py's equivalent check on HybridModel)."""
    cfg = DiffusionConfig(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
        cf_anchor_weight=0.15,
    )
    model = DiffPO(MODEL_CFG, cfg)
    x, a, y_fac, y_cf = _batch()
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])

    captured = {}
    original = model.calculate_diffusion_loss

    def spy(eps, eps_pred, gradient_mask, x_arg, a_arg, propnet):
        captured["gradient_mask"] = gradient_mask
        return original(eps, eps_pred, gradient_mask, x_arg, a_arg, propnet)

    model.calculate_diffusion_loss = spy
    model.compute_loss(x, a, y_fac, y_cf, pop_means=(3.0, -2.0))

    factual_mask = torch.stack([1 - a, a], dim=1)
    expected = factual_mask + 0.15 * (1.0 - factual_mask)
    assert torch.allclose(captured["gradient_mask"], expected)


def test_backward():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    model.total_loss(model.compute_loss(*_batch())).backward()
    assert model.denoiser.cond_proj.weight.grad is not None


def test_sample_outcomes_shapes():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y0, y1 = model.sample_outcomes(x, a, K=3)
    assert y0.shape == (B, 3)
    assert y1.shape == (B, 3)
    assert torch.isfinite(y0).all()


def test_sample_outcomes_clip_val_bounds_output():
    """clip_val must actually clip -- verified against a schedule that reliably diverges."""
    steep_diff_cfg = DiffusionConfig(
        num_steps=50,
        beta_start=0.0001,
        beta_end=0.5,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
    )
    torch.manual_seed(0)
    model = DiffPO(MODEL_CFG, steep_diff_cfg)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()

    torch.manual_seed(1)
    y0_unclipped, y1_unclipped = model.sample_outcomes(x, a, K=3, clip_val=None)
    unclipped_max = torch.max(y0_unclipped.abs().max(), y1_unclipped.abs().max())
    assert unclipped_max > 6.0, "test schedule should reliably diverge without clip_val"

    torch.manual_seed(1)
    y0_clipped, y1_clipped = model.sample_outcomes(x, a, K=3, clip_val=6.0)
    assert y0_clipped.abs().max() <= 6.0
    assert y1_clipped.abs().max() <= 6.0
    assert torch.isfinite(y0_clipped).all()
    assert torch.isfinite(y1_clipped).all()

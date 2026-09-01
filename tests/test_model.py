import pytest
import torch

from src.config import DiffusionConfig, ModelConfig
from src.model import HybridModel

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


def test_loss_component_keys_and_shapes():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    assert set(comps.keys()) == {
        "log_px",
        "log_pa",
        "kl",
        "diffusion_loss",
        "log_ry",
    }
    for k, v in comps.items():
        assert v.shape == (), f"{k} not scalar"


def test_loss_components_finite():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    for k, v in comps.items():
        assert torch.isfinite(v), f"{k} = {v}"


def test_total_loss_finite():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    loss = model.total_loss(model.compute_loss(*_batch()))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_backward():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    loss = model.total_loss(model.compute_loss(*_batch()))
    loss.backward()
    assert model.encoder.trunk[0].weight.grad is not None
    assert model.denoiser.cond_proj.weight.grad is not None


def test_sample_outcomes_shapes():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    K = 3
    a = torch.randint(0, 2, (B,)).float()
    y0, y1 = model.sample_outcomes(torch.randn(B, F), a, K=K)
    assert y0.shape == (B, K)
    assert y1.shape == (B, K)
    assert torch.isfinite(y0).all()
    assert torch.isfinite(y1).all()


# ── consistency loss ────────────────────────────────────────────────────────


def _consistency_cfg(**overrides):
    return DiffusionConfig(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
        **overrides,
    )


def test_consistency_loss_present_and_finite():
    cfg = _consistency_cfg(consistency_weight=1.0, consistency_min_tau_frac=0.0)
    model = HybridModel(MODEL_CFG, cfg)
    comps = model.compute_loss(*_batch(), epoch_frac=1.0)
    assert "consistency_loss" in comps
    assert "consistency_raw" in comps
    assert comps["consistency_loss"].shape == ()
    assert comps["consistency_raw"].shape == ()
    assert torch.isfinite(comps["consistency_loss"])
    assert torch.isfinite(comps["consistency_raw"])


def test_consistency_loss_detached_from_aux_outcome():
    cfg = _consistency_cfg(consistency_weight=1.0, consistency_min_tau_frac=0.0)
    model = HybridModel(MODEL_CFG, cfg)
    comps = model.compute_loss(*_batch(), epoch_frac=1.0)
    comps["consistency_loss"].backward()
    assert model.denoiser.cond_proj.weight.grad is not None
    assert model.aux_outcome.trunk[0].weight.grad is None


def test_consistency_loss_inert_by_default():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    assert "consistency_loss" not in comps
    assert "consistency_raw" not in comps


def test_consistency_loss_ramp():
    cfg = _consistency_cfg(
        consistency_weight=0.4, consistency_warmup_frac=0.5, consistency_min_tau_frac=0.0
    )
    model = HybridModel(MODEL_CFG, cfg)

    comps_start = model.compute_loss(*_batch(), epoch_frac=0.0)
    assert comps_start["consistency_loss"].item() == 0.0

    comps_end = model.compute_loss(*_batch(), epoch_frac=1.0)
    assert comps_end["consistency_loss"].item() == pytest.approx(
        0.4 * comps_end["consistency_raw"].item()
    )


def test_consistency_loss_fully_masked_is_zero():
    cfg = _consistency_cfg(consistency_weight=1.0, consistency_min_tau_frac=1.0)
    model = HybridModel(MODEL_CFG, cfg)
    comps = model.compute_loss(*_batch(), epoch_frac=1.0)
    assert torch.isfinite(comps["consistency_loss"])
    assert comps["consistency_loss"].item() == 0.0


# ── cf population-mean anchor ───────────────────────────────────────────────


def _anchor_cfg(**overrides):
    return DiffusionConfig(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
        **overrides,
    )


def test_cf_anchor_inert_by_default():
    model = HybridModel(MODEL_CFG, DIFF_CFG)
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
    cf_target compute_loss actually constructed, not just a property of two tensors built
    separately from the test. A wrong or skipped substitution would fail this."""
    cfg = _anchor_cfg(cf_anchor_weight=0.1)
    model = HybridModel(MODEL_CFG, cfg)
    x = torch.randn(B, 5)
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


def test_cf_anchor_soft_mask_weight():
    cfg = _anchor_cfg(cf_anchor_weight=0.15)
    model = HybridModel(MODEL_CFG, cfg)
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])
    factual_mask = torch.stack([1 - a, a], dim=1)
    soft_mask = factual_mask + model._cf_anchor_weight * (1.0 - factual_mask)
    expected = torch.tensor([[1.0, 0.15], [0.15, 1.0], [1.0, 0.15], [0.15, 1.0]])
    assert torch.allclose(soft_mask, expected)


def test_cf_anchor_finite_loss():
    cfg = _anchor_cfg(cf_anchor_weight=0.1)
    model = HybridModel(MODEL_CFG, cfg)
    x, a, y_fac, y_cf = _batch()
    comps = model.compute_loss(x, a, y_fac, y_cf, pop_means=(3.0, -2.0))
    assert torch.isfinite(comps["diffusion_loss"])


def test_apply_cf_anchor_truth_table():
    cfg = _anchor_cfg(cf_anchor_weight=0.1)
    model_on = HybridModel(MODEL_CFG, cfg)
    model_off = HybridModel(MODEL_CFG, DIFF_CFG)  # cf_anchor_weight defaults to 0.0
    a = torch.tensor([0.0, 1.0])
    y_cf = torch.tensor([5.0, 6.0])

    _, active = model_on._apply_cf_anchor(a, y_cf, pop_means=(1.0, 2.0))
    assert active is True

    _, active = model_on._apply_cf_anchor(a, y_cf, pop_means=None)
    assert active is False

    _, active = model_off._apply_cf_anchor(a, y_cf, pop_means=(1.0, 2.0))
    assert active is False

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

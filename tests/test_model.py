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
    model = HybridModel(MODEL_CFG, DIFF_CFG)  # cf_anchor_weight defaults to 0.0
    x, a, y_fac, y_cf = _batch()
    torch.manual_seed(0)
    comps = model.compute_loss(x, a, y_fac, y_cf)
    cf_target, anchor_active = model._apply_cf_anchor(x, a, y_cf)
    assert anchor_active is False
    assert torch.equal(cf_target, y_cf)
    assert torch.isfinite(comps["diffusion_loss"])


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
    comps = model.compute_loss(x, a, y_fac, y_cf)
    assert torch.isfinite(comps["diffusion_loss"])


def test_cf_anchor_uses_aux_outcome_not_y_cf():
    """Spy on the real _noise_targets call inside compute_loss -- directly verifies what
    cf_target compute_loss actually constructed. A wrong or skipped substitution fails this."""
    cfg = _anchor_cfg(cf_anchor_weight=0.1)
    model = HybridModel(MODEL_CFG, cfg)
    x = torch.randn(B, 5)
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y_fac = torch.randn(B)
    y_cf = torch.full((B,), 999.0)  # deliberately extreme -- must NOT reach _noise_targets

    with torch.no_grad():
        expected_cf_target = model.aux_outcome.mean(x, 1.0 - a)

    captured = {}
    original = model._noise_targets

    def spy(batch_size, device, a_arg, y_fac_arg, y_cf_arg):
        captured["y_cf_arg"] = y_cf_arg
        return original(batch_size, device, a_arg, y_fac_arg, y_cf_arg)

    model._noise_targets = spy
    model.compute_loss(x, a, y_fac, y_cf)

    assert torch.allclose(captured["y_cf_arg"], expected_cf_target)
    assert not torch.allclose(captured["y_cf_arg"], y_cf)


def test_cf_anchor_softens_gradient_mask_in_compute_loss():
    """Spy on _noise_targets (for eps/factual_mask) and the denoiser (for eps_pred) --
    verifies the ACTUAL tensors reaching compute_loss's diffusion_loss computation, not
    just a standalone recomputation of the formula."""
    cfg = _anchor_cfg(cf_anchor_weight=0.15)
    model = HybridModel(MODEL_CFG, cfg)
    x, a, y_fac, y_cf = _batch()
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])

    captured = {}
    original_noise_targets = model._noise_targets

    def noise_spy(batch_size, device, a_arg, y_fac_arg, y_cf_arg):
        noisy_y, tau, eps, factual_mask = original_noise_targets(
            batch_size, device, a_arg, y_fac_arg, y_cf_arg
        )
        captured["eps"] = eps
        captured["factual_mask"] = factual_mask
        return noisy_y, tau, eps, factual_mask

    model._noise_targets = noise_spy

    original_denoiser_forward = model.denoiser.forward

    def denoiser_spy(*args, **kwargs):
        eps_pred = original_denoiser_forward(*args, **kwargs)
        captured["eps_pred"] = eps_pred
        return eps_pred

    model.denoiser.forward = denoiser_spy

    comps = model.compute_loss(x, a, y_fac, y_cf)

    factual_mask = captured["factual_mask"]
    expected_mask = factual_mask + 0.15 * (1.0 - factual_mask)
    expected_loss = (
        (((captured["eps_pred"] - captured["eps"]) * expected_mask) ** 2).sum(dim=1).mean()
    )
    assert torch.allclose(comps["diffusion_loss"], expected_loss)


def test_cf_anchor_detached_from_aux_outcome():
    """Backward on diffusion_loss alone (not total_loss, which also legitimately trains
    aux_outcome via log_ry and would confound the check). If detachment were dropped,
    cf_target's dependency on aux_outcome.mean(x, 1-a) would make this fail."""
    cfg = _anchor_cfg(cf_anchor_weight=1.0)
    model = HybridModel(MODEL_CFG, cfg)
    comps = model.compute_loss(*_batch())
    comps["diffusion_loss"].backward()
    assert model.denoiser.cond_proj.weight.grad is not None
    assert model.aux_outcome.trunk[0].weight.grad is None

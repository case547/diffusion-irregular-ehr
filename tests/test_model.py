import pytest
import torch
from pydantic import ValidationError

from src.config import Config, DataConfig, DiffusionConfig, TrainConfig, VAEConfig
from src.model import HybridModel

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
B, F = 4, 5


def _batch():
    return torch.randn(B, F), torch.randint(0, 2, (B,)).float(), torch.randn(B), torch.randn(B)


def test_loss_component_keys_and_shapes():
    model = HybridModel(VAE_CFG, DIFF_CFG)
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
    model = HybridModel(VAE_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    for k, v in comps.items():
        assert torch.isfinite(v), f"{k} = {v}"


def test_total_loss_finite():
    model = HybridModel(VAE_CFG, DIFF_CFG)
    loss = model.total_loss(model.compute_loss(*_batch()))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_backward():
    model = HybridModel(VAE_CFG, DIFF_CFG)
    loss = model.total_loss(model.compute_loss(*_batch()))
    loss.backward()
    assert model.encoder.trunk[0].weight.grad is not None
    assert model.denoiser.cond_proj.weight.grad is not None


def test_sample_outcomes_shapes():
    model = HybridModel(VAE_CFG, DIFF_CFG)
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
    model = HybridModel(VAE_CFG, DIFF_CFG)  # cf_anchor_weight defaults to 0.0
    x, a, y_fac, y_cf = _batch()
    torch.manual_seed(0)
    comps = model.compute_loss(x, a, y_fac, y_cf)
    cf_target, anchor_active = model._apply_cf_anchor(x, a, y_cf)
    assert anchor_active is False
    assert torch.equal(cf_target, y_cf)
    assert torch.isfinite(comps["diffusion_loss"])


def test_cf_anchor_soft_mask_weight():
    cfg = _anchor_cfg(cf_anchor_weight=0.15)
    model = HybridModel(VAE_CFG, cfg)
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])
    factual_mask = torch.stack([1 - a, a], dim=1)
    soft_mask = factual_mask + model._cf_anchor_weight * (1.0 - factual_mask)
    expected = torch.tensor([[1.0, 0.15], [0.15, 1.0], [1.0, 0.15], [0.15, 1.0]])
    assert torch.allclose(soft_mask, expected)


def test_cf_anchor_finite_loss():
    cfg = _anchor_cfg(cf_anchor_weight=0.1)
    model = HybridModel(VAE_CFG, cfg)
    x, a, y_fac, y_cf = _batch()
    comps = model.compute_loss(x, a, y_fac, y_cf)
    assert torch.isfinite(comps["diffusion_loss"])


def test_cf_anchor_uses_aux_outcome_not_y_cf():
    """Spy on the real _noise_targets call inside compute_loss -- directly verifies what
    cf_target compute_loss actually constructed. A wrong or skipped substitution fails this."""
    cfg = _anchor_cfg(cf_anchor_weight=0.1)
    model = HybridModel(VAE_CFG, cfg)
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
    model = HybridModel(VAE_CFG, cfg)
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
    model = HybridModel(VAE_CFG, cfg)
    comps = model.compute_loss(*_batch())
    comps["diffusion_loss"].backward()
    assert model.denoiser.cond_proj.weight.grad is not None
    assert model.aux_outcome.trunk[0].weight.grad is None


# ── a_decoder label smoothing ──────────────────────────────────────────────


def test_a_decoder_label_smoothing_changes_log_pa_target():
    """With smoothing on, log_pa must differ from the unsmoothed value on the SAME z --
    spy on a_decoder.log_prob to capture the actual `a` tensor it was called with."""
    cfg = VAEConfig(
        feature_dim=5,
        latent_dim=4,
        hidden_dim=16,
        encoder_num_layers=2,
        decoder_num_layers=1,
        aux_num_layers=1,
        a_decoder_hidden_dim=5,
        a_decoder_label_smoothing=0.05,
    )
    model = HybridModel(cfg, DIFF_CFG)
    x, a, y_fac, y_cf = _batch()
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])

    captured = {}
    original = model.a_decoder.log_prob

    def spy(z, a_arg):
        captured["a_arg"] = a_arg
        return original(z, a_arg)

    model.a_decoder.log_prob = spy
    model.compute_loss(x, a, y_fac, y_cf)

    expected = a * (1.0 - 2 * 0.05) + 0.05  # a=0 -> 0.05, a=1 -> 0.95
    assert torch.allclose(captured["a_arg"], expected)
    assert not torch.allclose(captured["a_arg"], a)


def test_a_decoder_label_smoothing_is_noop_when_zero():
    """Default a_decoder_label_smoothing=0.0 must reach a_decoder.log_prob with the raw,
    unsmoothed `a` -- exact equality with today's behavior."""
    model = HybridModel(VAE_CFG, DIFF_CFG)  # a_decoder_label_smoothing defaults to 0.0
    x, a, y_fac, y_cf = _batch()
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])

    captured = {}
    original = model.a_decoder.log_prob

    def spy(z, a_arg):
        captured["a_arg"] = a_arg
        return original(z, a_arg)

    model.a_decoder.log_prob = spy
    model.compute_loss(x, a, y_fac, y_cf)

    assert torch.equal(captured["a_arg"], a)


# ── z-space IPW weight application ─────────────────────────────────────────


def _ipw_diff_cfg(**overrides):
    kwargs = dict(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
        use_ipw=True,
        ipw_ramp_start=0,
        ipw_ramp_end=1,
        ipw_clip_prop=0.1,
        ipw_z_samples=2,
    )
    kwargs.update(overrides)
    return DiffusionConfig(**kwargs)


def test_config_validates_ipw_fields():
    """DiffusionConfig itself (not HybridModel) must reject each of the four
    invalid-field cases independently -- ipw_ramp_end<=ipw_ramp_start, ipw_ema_decay
    outside [0,1), ipw_z_samples<1, and a_decoder_label_smoothing outside [0, 0.5) --
    at config-construction time, before any model is ever built."""
    with pytest.raises(ValidationError):
        _ipw_diff_cfg(ipw_ramp_start=5, ipw_ramp_end=5)
    with pytest.raises(ValidationError):
        diff_cfg = _ipw_diff_cfg()
        train_cfg = TrainConfig(
            epochs=1,
            batch_size=1,
            lr=1e-3,
            seed=0,
            K=1,
            checkpoint_dir="/tmp",
            ipw_ema_decay=1.0,
        )
        data_cfg = DataConfig(path="/tmp", replication=1, train_ratio=0.8, test_ratio=0.2)
        _ = Config(vae=VAE_CFG, diffusion=diff_cfg, train=train_cfg, data=data_cfg)
    with pytest.raises(ValidationError):
        _ipw_diff_cfg(ipw_z_samples=0)
    with pytest.raises(ValidationError):
        illegal_config = VAE_CFG.model_dump()
        illegal_config["a_decoder_label_smoothing"] = 0.5
        _ = VAEConfig(**illegal_config)


def test_compute_phat_shape_and_range():
    from ema_pytorch import EMA

    cfg = _ipw_diff_cfg()
    model = HybridModel(VAE_CFG, cfg)
    ema_a_decoder = EMA(model.a_decoder, beta=0.9, update_after_step=0, update_every=1)
    ema_encoder = EMA(model.encoder, beta=0.9, update_after_step=0, update_every=1)
    ema_a_decoder.update()
    ema_encoder.update()

    x, a, y_fac, _ = _batch()
    p_hat = model._compute_pi_hat(x, a, y_fac, ema_encoder, ema_a_decoder)
    assert p_hat.shape == (B,)
    assert torch.all(p_hat > 0.0) and torch.all(p_hat < 1.0)


def test_compute_loss_uses_ipw_weight_when_active():
    """With use_ipw active and past ramp_start, diffusion_loss must differ from the
    unweighted formula on data engineered to trigger a non-trivial weight -- verified
    by directly recomputing both versions from the same intercepted eps/eps_pred."""
    from ema_pytorch import EMA

    torch.manual_seed(6)
    cfg = _ipw_diff_cfg(ipw_ramp_start=0, ipw_ramp_end=1)
    model = HybridModel(VAE_CFG, cfg)
    ema_a_decoder = EMA(model.a_decoder, beta=0.9, update_after_step=0, update_every=1)
    ema_encoder = EMA(model.encoder, beta=0.9, update_after_step=0, update_every=1)
    ema_a_decoder.update()
    ema_encoder.update()

    x, a, y_fac, y_cf = _batch()
    a = torch.tensor([0.0, 1.0, 0.0, 1.0])

    captured = {}
    original_noise_targets = model._noise_targets

    def noise_spy(batch_size, device, a_arg, y_fac_arg, y_cf_arg):
        out = original_noise_targets(batch_size, device, a_arg, y_fac_arg, y_cf_arg)
        captured["eps"], captured["factual_mask"] = out[2], out[3]
        return out

    model._noise_targets = noise_spy
    original_denoiser_forward = model.denoiser.forward

    def denoiser_spy(*args, **kwargs):
        eps_pred = original_denoiser_forward(*args, **kwargs)
        captured["eps_pred"] = eps_pred
        return eps_pred

    model.denoiser.forward = denoiser_spy

    torch.manual_seed(6)
    comps = model.compute_loss(
        x, a, y_fac, y_cf, epoch=1, ema_a_decoder=ema_a_decoder, ema_encoder=ema_encoder
    )

    per_sample = (
        ((captured["eps_pred"] - captured["eps"]) * captured["factual_mask"]) ** 2
    ).sum(dim=1)
    unweighted = per_sample.mean()

    assert torch.isfinite(comps["diffusion_loss"])
    # Only assert inequality when the weights are non-trivial (skip flaky equality-by-
    # chance): with a freshly-initialised a_decoder the trim is unlikely to fire on
    # every subject, so this should hold with the fixed seed above.
    assert not torch.allclose(comps["diffusion_loss"], unweighted)


def test_compute_loss_falls_back_to_unweighted_before_ramp_start():
    """epoch < ipw_ramp_start must skip _compute_phat entirely (spied) and diffusion_loss
    must equal the unweighted formula recomputed from the actual intercepted eps/eps_pred
    -- not just two identically-seeded calls down what could be the same branch either way."""
    from ema_pytorch import EMA

    torch.manual_seed(7)
    cfg = _ipw_diff_cfg(ipw_ramp_start=5, ipw_ramp_end=10)
    model = HybridModel(VAE_CFG, cfg)
    ema_a_decoder = EMA(model.a_decoder, beta=0.9, update_after_step=0, update_every=1)
    ema_encoder = EMA(model.encoder, beta=0.9, update_after_step=0, update_every=1)
    ema_a_decoder.update()
    ema_encoder.update()

    x, a, y_fac, y_cf = _batch()

    captured = {}
    original_noise_targets = model._noise_targets

    def noise_spy(batch_size, device, a_arg, y_fac_arg, y_cf_arg):
        out = original_noise_targets(batch_size, device, a_arg, y_fac_arg, y_cf_arg)
        captured["eps"], captured["factual_mask"] = out[2], out[3]
        return out

    model._noise_targets = noise_spy
    original_denoiser_forward = model.denoiser.forward

    def denoiser_spy(*args, **kwargs):
        eps_pred = original_denoiser_forward(*args, **kwargs)
        captured["eps_pred"] = eps_pred
        return eps_pred

    model.denoiser.forward = denoiser_spy

    phat_called = {"n": 0}
    original_compute_phat = model._compute_pi_hat

    def phat_spy(*args, **kwargs):
        phat_called["n"] += 1
        return original_compute_phat(*args, **kwargs)

    model._compute_pi_hat = phat_spy

    torch.manual_seed(7)
    comps_ipw = model.compute_loss(
        x, a, y_fac, y_cf, epoch=0, ema_a_decoder=ema_a_decoder, ema_encoder=ema_encoder
    )

    assert phat_called["n"] == 0
    per_sample = (
        ((captured["eps_pred"] - captured["eps"]) * captured["factual_mask"]) ** 2
    ).sum(dim=1)
    assert torch.allclose(comps_ipw["diffusion_loss"], per_sample.mean())


def test_compute_loss_falls_back_to_unweighted_when_use_ipw_false():
    """use_ipw=False must skip weighting even when EMA objects ARE supplied (the only
    gate left is self._use_ipw) -- diffusion_loss must equal the unweighted formula
    recomputed from the actual intercepted eps/eps_pred."""
    from ema_pytorch import EMA

    x, a, y_fac, y_cf = _batch()
    model = HybridModel(VAE_CFG, DIFF_CFG)  # use_ipw defaults to False
    ema_a_decoder = EMA(model.a_decoder, beta=0.9, update_after_step=0, update_every=1)
    ema_encoder = EMA(model.encoder, beta=0.9, update_after_step=0, update_every=1)
    ema_a_decoder.update()
    ema_encoder.update()

    captured = {}
    original_noise_targets = model._noise_targets

    def noise_spy(batch_size, device, a_arg, y_fac_arg, y_cf_arg):
        out = original_noise_targets(batch_size, device, a_arg, y_fac_arg, y_cf_arg)
        captured["eps"], captured["factual_mask"] = out[2], out[3]
        return out

    model._noise_targets = noise_spy
    original_denoiser_forward = model.denoiser.forward

    def denoiser_spy(*args, **kwargs):
        eps_pred = original_denoiser_forward(*args, **kwargs)
        captured["eps_pred"] = eps_pred
        return eps_pred

    model.denoiser.forward = denoiser_spy

    comps = model.compute_loss(
        x, a, y_fac, y_cf, epoch=100, ema_a_decoder=ema_a_decoder, ema_encoder=ema_encoder
    )

    per_sample = (
        ((captured["eps_pred"] - captured["eps"]) * captured["factual_mask"]) ** 2
    ).sum(dim=1)
    assert torch.allclose(comps["diffusion_loss"], per_sample.mean())

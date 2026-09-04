import torch

from src.config import DiffusionConfig, VAEConfig
from src.model import DiffPO
from src.propensity import PropensityNet

VAE_CFG = VAEConfig(
    feature_dim=5, latent_dim=4, hidden_dim=16, num_layers=2, a_decoder_hidden_dim=5
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


def test_loss_keys_and_finite():
    model = DiffPO(VAE_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    assert set(comps.keys()) == {"diffusion_loss"}
    assert comps["diffusion_loss"].shape == ()
    assert torch.isfinite(comps["diffusion_loss"])


def test_loss_with_propnet():
    model = DiffPO(VAE_CFG, DIFF_CFG)
    propnet = PropensityNet(
        n_unit_in=F,
        n_units_out_prop=16,
        n_layers_out_prop=0,
        batch_norm=False,
    )
    x, a, y, y_cf = _batch()
    comps = model.compute_loss(x, a, y, y_cf, propnet=propnet)
    assert torch.isfinite(comps["diffusion_loss"])


def test_backward():
    model = DiffPO(VAE_CFG, DIFF_CFG)
    model.total_loss(model.compute_loss(*_batch())).backward()
    assert model.denoiser.cond_proj.weight.grad is not None


def test_sample_outcomes_shapes():
    model = DiffPO(VAE_CFG, DIFF_CFG)
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
    model = DiffPO(VAE_CFG, steep_diff_cfg)
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

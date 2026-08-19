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


def test_ddpm_reverse_without_log_trajectory_returns_tensor():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y = model._ddpm_reverse(B, x, a, torch.device("cpu"))
    assert isinstance(y, torch.Tensor)
    assert y.shape == (B, 2)


def test_ddpm_reverse_log_trajectory_shapes():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y_final, y_traj, eps_traj = model._ddpm_reverse(
        B, x, a, torch.device("cpu"), log_trajectory=True
    )
    assert y_final.shape == (B, 2)
    assert y_traj.shape == (DIFF_CFG.num_steps, B, 2)
    assert eps_traj.shape == (DIFF_CFG.num_steps, B, 2)
    assert torch.isfinite(y_final).all()
    assert torch.isfinite(y_traj).all()
    assert torch.isfinite(eps_traj).all()


def test_sample_ddim_shapes():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y = model.sample_ddim(x, a)
    assert y.shape == (B, 2)
    assert torch.isfinite(y).all()


def test_sample_ddim_deterministic_with_shared_y_init():
    """The whole confounding-diffusion-analysis notebook depends on this: two separate
    sample_ddim calls given the same y_init must produce bit-identical trajectories, so
    that any difference between two models' outputs is due to the models, not RNG luck."""
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y_init = torch.randn(B, 2)

    y1 = model.sample_ddim(x, a, y_init=y_init)
    y2 = model.sample_ddim(x, a, y_init=y_init)
    assert torch.equal(y1, y2)


def test_sample_ddim_log_trajectory_shapes():
    model = DiffPO(MODEL_CFG, DIFF_CFG)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y, y_traj, eps_traj = model.sample_ddim(x, a, log_trajectory=True)
    assert y.shape == (B, 2)
    assert y_traj.shape == (DIFF_CFG.num_steps, B, 2)
    assert eps_traj.shape == (DIFF_CFG.num_steps, B, 2)
    assert torch.isfinite(y).all()

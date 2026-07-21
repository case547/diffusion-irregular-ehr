import torch
from src.config import ModelConfig, DiffusionConfig
from src.model import DiffPOCEVAE

MODEL_CFG = ModelConfig(feature_dim=5, latent_dim=4, hidden_dim=16, num_layers=2)
DIFF_CFG = DiffusionConfig(
    num_steps=10, beta_start=0.0001, beta_end=0.02,
    schedule="quad", embedding_dim=16, block_dim=16, hidden_dim=32, num_blocks=2,
)
B, F = 4, 5


def _batch():
    return torch.randn(B, F), torch.randint(0, 2, (B,)).float(), torch.randn(B), torch.randn(B)


def test_loss_component_keys_and_shapes():
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    assert set(comps.keys()) == {"log_px", "log_pa", "kl", "diffusion_loss", "log_qa", "log_qy"}
    for k, v in comps.items():
        assert v.shape == (), f"{k} not scalar"


def test_loss_components_finite():
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    comps = model.compute_loss(*_batch())
    for k, v in comps.items():
        assert torch.isfinite(v), f"{k} = {v}"


def test_total_loss_finite():
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    loss = model.total_loss(model.compute_loss(*_batch()))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_backward():
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    loss = model.total_loss(model.compute_loss(*_batch()))
    loss.backward()
    assert model.encoder.trunk[0].weight.grad is not None
    assert model.denoiser.cond_proj.weight.grad is not None


def test_sample_outcomes_shapes():
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    K = 3
    a = torch.randint(0, 2, (B,)).float()
    y0, y1 = model.sample_outcomes(torch.randn(B, F), a, K=K)
    assert y0.shape == (B, K)
    assert y1.shape == (B, K)
    assert torch.isfinite(y0).all()
    assert torch.isfinite(y1).all()

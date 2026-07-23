"""Integration tests: one training step, loss decreases, checkpoint saved."""

import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import Config, DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from src.data import CausalDataset
from src.model import DiffPOCEVAE
from train import calculate_val_loss

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


def _loader(n: int = 64, f: int = 5, batch_size: int = 16):
    ds = CausalDataset(
        np.random.randn(n, f).astype(np.float32),
        np.random.randint(0, 2, n).astype(np.float32),
        np.random.randn(n).astype(np.float32),
        np.random.randn(n).astype(np.float32),  # y_cf
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_one_training_step():
    torch.manual_seed(0)
    np.random.seed(0)
    loader = _loader()
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    opt = Adam(model.parameters(), lr=1e-3)
    model.train()
    batch = next(iter(loader))
    opt.zero_grad()
    loss = model.total_loss(
        model.compute_loss(batch["x"], batch["a"], batch["y"], batch["y_cf"])
    )
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_loss_decreases_over_20_steps():
    torch.manual_seed(1)
    np.random.seed(1)
    loader = _loader(n=64, batch_size=64)
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    opt = Adam(model.parameters(), lr=1e-2)
    model.train()
    losses = []
    batch = next(iter(loader))
    x, a, y, y_cf = batch["x"], batch["a"], batch["y"], batch["y_cf"]
    for _ in range(20):
        opt.zero_grad()
        loss = model.total_loss(model.compute_loss(x, a, y, y_cf))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


def test_val_loss_finite():
    torch.manual_seed(2)
    model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG)
    loader = _loader()
    device = torch.device("cpu")
    comps = calculate_val_loss(model, loader, device)
    assert set(comps.keys()) == {
        "log_px",
        "log_pa",
        "kl",
        "diffusion_loss",
        "log_qa",
        "log_qy",
        "total_loss",
    }
    assert all(np.isfinite(v) for v in comps.values())


def test_checkpoint_saved(tmp_path):
    torch.manual_seed(2)
    np.random.seed(2)
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DIFF_CFG,
        train=TrainConfig(
            epochs=4,
            batch_size=16,
            lr=1e-3,
            seed=2,
            K=2,
            patience=10,
            warmup_epochs=0,
            checkpoint_dir=str(tmp_path),
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    from train import _train_loop

    loader = _loader(n=32, f=5, batch_size=16)
    model = DiffPOCEVAE(cfg.model, cfg.diffusion)
    device = torch.device("cpu")
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "best_model.pth")
    _train_loop(model, loader, loader, cfg, device, ckpt_path)
    assert os.path.exists(ckpt_path)
    model2 = DiffPOCEVAE(cfg.model, cfg.diffusion)
    model2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))  # must not raise

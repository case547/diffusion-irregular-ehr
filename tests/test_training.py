"""Integration tests: one training step, loss decreases, checkpoint saved."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.auxiliary import AuxOutcome
from src.config import Config, DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from src.data import CausalDataset
from src.model import HybridModel
from train import calculate_val_loss, train_aux_outcome

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


def _install_ema_capture_hook(captured: dict) -> None:
    """Monkeypatch ema_pytorch.EMA (as imported into train.py) so the test can grab
    the actual instances _train_loop constructs, without _train_loop needing to
    return or expose them itself."""
    import train

    original_ema = train.EMA

    class _CapturingEMA(original_ema):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            key = "ema_a_decoder" if "ema_a_decoder" not in captured else "ema_encoder"
            captured[key] = self

    train.EMA = _CapturingEMA


def test_one_training_step():
    torch.manual_seed(0)
    np.random.seed(0)
    loader = _loader()
    model = HybridModel(MODEL_CFG, DIFF_CFG)
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
    model = HybridModel(MODEL_CFG, DIFF_CFG)
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
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    loader = _loader()
    device = torch.device("cpu")
    comps = calculate_val_loss(model, loader, device)
    assert set(comps.keys()) == {
        "log_px",
        "log_pa",
        "kl",
        "diffusion_loss",
        "log_ry",
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
            epochs=4, batch_size=16, lr=1e-3, seed=2, K=2, checkpoint_dir=str(tmp_path)
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    from train import _train_loop

    loader = _loader(n=32, f=5, batch_size=16)
    model = HybridModel(cfg.model, cfg.diffusion)
    device = torch.device("cpu")
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    run_id = "pytest_run"
    _train_loop(model, loader, loader, cfg, device, run_id)
    ckpt_path = Path(cfg.train.checkpoint_dir) / f"final_model_{run_id}.pth"
    assert ckpt_path.exists()
    model2 = HybridModel(cfg.model, cfg.diffusion)
    model2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))  # must not raise


def _aux_loaders(n: int = 64, f: int = 5, batch_size: int = 16):
    """Train/val loaders with DIFFERENT underlying data -- val loss should plateau or
    worsen quickly on random data unrelated to train, giving early stopping something
    real to trigger on."""
    train_ds = CausalDataset(
        np.random.randn(n, f).astype(np.float32),
        np.random.randint(0, 2, n).astype(np.float32),
        np.random.randn(n).astype(np.float32),
    )
    val_ds = CausalDataset(
        np.random.randn(n, f).astype(np.float32),
        np.random.randint(0, 2, n).astype(np.float32),
        np.random.randn(n).astype(np.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def test_train_aux_outcome_loss_decreases():
    torch.manual_seed(0)
    np.random.seed(0)
    train_loader, val_loader = _aux_loaders()
    aux = AuxOutcome(MODEL_CFG.feature_dim, MODEL_CFG.hidden_dim, MODEL_CFG.num_layers)
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DIFF_CFG,
        train=TrainConfig(
            epochs=20, batch_size=16, lr=1e-2, seed=0, K=2, checkpoint_dir="/tmp"
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    device = torch.device("cpu")

    aux.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        x, a, y = batch["x"], batch["a"], batch["y"]
        loss_before = -aux.log_prob(x, a, y).mean().item()

    train_aux_outcome(aux, train_loader, val_loader, cfg, device, patience=20, min_epochs=20)

    aux.eval()
    with torch.no_grad():
        loss_after = -aux.log_prob(x, a, y).mean().item()
    assert loss_after < loss_before


def test_train_aux_outcome_early_stopping_fires():
    """patience=1, min_epochs=1, and a val set unrelated to train -- val loss should fail
    to improve almost immediately, so training must halt well before cfg.train.epochs."""
    torch.manual_seed(1)
    np.random.seed(1)
    train_loader, val_loader = _aux_loaders()
    aux = AuxOutcome(MODEL_CFG.feature_dim, MODEL_CFG.hidden_dim, MODEL_CFG.num_layers)
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DIFF_CFG,
        train=TrainConfig(
            epochs=50, batch_size=16, lr=1e-2, seed=1, K=2, checkpoint_dir="/tmp"
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    device = torch.device("cpu")

    epochs_logged = []
    train_aux_outcome(
        aux,
        train_loader,
        val_loader,
        cfg,
        device,
        log_fn=lambda d, step: epochs_logged.append(step),
        patience=1,
        min_epochs=1,
    )
    assert len(epochs_logged) < 50, "training ran to completion instead of stopping early"


def test_train_aux_outcome_restores_best_state():
    """Tiny patience so the best snapshot is taken early and training continues a couple
    more epochs before stopping -- assert the FINAL weights match the best-epoch snapshot,
    not whatever the last (worse) epoch trained to."""
    torch.manual_seed(2)
    np.random.seed(2)
    train_loader, val_loader = _aux_loaders()
    aux = AuxOutcome(MODEL_CFG.feature_dim, MODEL_CFG.hidden_dim, MODEL_CFG.num_layers)
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DIFF_CFG,
        train=TrainConfig(
            epochs=50, batch_size=16, lr=1e-1, seed=2, K=2, checkpoint_dir="/tmp"
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    device = torch.device("cpu")

    val_losses = []

    train_aux_outcome(
        aux,
        train_loader,
        val_loader,
        cfg,
        device,
        log_fn=lambda d, step: val_losses.append(d["pretrain_aux/val_nll"]),
        patience=2,
        min_epochs=1,
    )
    best_val_loss_seen = min(val_losses)

    aux.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        x, a, y = batch["x"], batch["a"], batch["y"]
        final_val_loss = -aux.log_prob(x, a, y).mean().item()
    # the restored (best) state's val loss on this same batch should be at or near the
    # best value logged during training, not the (worse) value from a later epoch
    assert final_val_loss == pytest.approx(best_val_loss_seen, abs=0.5)


def test_train_loop_passes_current_epoch_to_compute_loss():
    torch.manual_seed(3)
    np.random.seed(3)
    loader = _loader(n=32, batch_size=16)
    model = HybridModel(MODEL_CFG, DIFF_CFG)
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DIFF_CFG,
        train=TrainConfig(
            epochs=3, batch_size=16, lr=1e-3, seed=3, K=2, checkpoint_dir="/tmp"
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    device = torch.device("cpu")

    seen_epochs = []
    original = model.compute_loss

    def spy(*args, **kwargs):
        seen_epochs.append(kwargs.get("epoch", args[5] if len(args) > 5 else 0))
        return original(*args, **kwargs)

    model.compute_loss = spy
    from train import _train_loop

    _train_loop(model, loader, loader, cfg, device, "pytest_epoch_run")
    # 2 batches/epoch (32/16) * 3 epochs of training calls, plus calculate_val_loss's
    # own calls each epoch -- every training-batch call for epoch e must report e.
    training_epochs_seen = sorted(set(seen_epochs))
    assert training_epochs_seen == [0, 1, 2]


def test_train_loop_builds_and_updates_ema_when_ipw_enabled():
    """ema_a_decoder/ema_encoder must exist as EMA-wrapped copies of the live modules
    and their internal step counter must have advanced once per training batch."""
    from ema_pytorch import EMA

    torch.manual_seed(4)
    np.random.seed(4)
    n, batch_size = 32, 16
    epochs = 2
    loader = _loader(n=n, batch_size=batch_size)
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DiffusionConfig(
            num_steps=10,
            beta_start=0.0001,
            beta_end=0.02,
            schedule="quad",
            embedding_dim=16,
            block_dim=16,
            hidden_dim=32,
            num_blocks=2,
            use_ipw=True,
            ipw_ramp_start=1,
            ipw_ramp_end=2,
            ipw_ema_decay=0.9,
        ),
        train=TrainConfig(
            epochs=epochs, batch_size=batch_size, lr=1e-3, seed=4, K=2, checkpoint_dir="/tmp"
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    model = HybridModel(cfg.model, cfg.diffusion)
    device = torch.device("cpu")

    captured: dict = {}
    _install_ema_capture_hook(captured)  # see Step 3 -- test-only introspection helper

    from train import _train_loop

    _train_loop(model, loader, loader, cfg, device, "pytest_ema_run")

    assert isinstance(captured.get("ema_a_decoder"), EMA)
    assert isinstance(captured.get("ema_encoder"), EMA)
    n_batches_per_epoch = n // batch_size  # 2
    expected_steps = n_batches_per_epoch * epochs  # 4
    assert captured["ema_a_decoder"].step.item() == expected_steps
    assert captured["ema_encoder"].step.item() == expected_steps


def test_train_loop_skips_ema_when_ipw_disabled():
    """use_ipw=False (the default) must not construct any EMA objects at all."""
    torch.manual_seed(5)
    np.random.seed(5)
    loader = _loader(n=32, batch_size=16)
    model = HybridModel(MODEL_CFG, DIFF_CFG)  # use_ipw defaults to False
    cfg = Config(
        model=MODEL_CFG,
        diffusion=DIFF_CFG,
        train=TrainConfig(
            epochs=1, batch_size=16, lr=1e-3, seed=5, K=2, checkpoint_dir="/tmp"
        ),
        data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
    )
    device = torch.device("cpu")

    captured: dict = {}
    _install_ema_capture_hook(captured)

    from train import _train_loop

    _train_loop(model, loader, loader, cfg, device, "pytest_no_ema_run")

    assert captured.get("ema_a_decoder") is None
    assert captured.get("ema_encoder") is None

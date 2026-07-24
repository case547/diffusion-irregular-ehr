"""Shared training utilities: val_loss, evaluate, _train_loop."""

import logging
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.config import Config
from src.metrics import coverage, pehe, rmse
from src.model import _DiffusionBase
from src.propensity import PropensityNet

logger = logging.getLogger(__name__)


def calculate_val_loss(
    model: _DiffusionBase,
    loader: DataLoader,
    device: torch.device,
    propnet: PropensityNet | None = None,
) -> dict[str, float]:
    """Mean of all loss components on loader. No sampling -- cheap forward pass only."""
    model.eval()
    totals: dict = defaultdict(float)
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            y = batch["y"].to(device)
            y_cf = batch["y_cf"].to(device)
            comps = model.compute_loss(x, a, y, y_cf, propnet=propnet)
            for k, v in comps.items():
                totals[k] += v.item()
            totals["total_loss"] += model.total_loss(comps).item()
            n += 1
    return {k: v / n for k, v in totals.items()}


def evaluate(
    model: _DiffusionBase, loader: DataLoader, K: int, device: torch.device
) -> dict[str, float]:
    """Test-time evaluation: generate K PO samples and compute coverage, RMSE, PEHE."""
    model.eval()
    all_y0, all_y1, all_mu0, all_mu1 = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            y0_s, y1_s = model.sample_outcomes(x, a, K=K)  # each (B,K)
            all_y0.append(y0_s.cpu())
            all_y1.append(y1_s.cpu())
            all_mu0.append(batch["mu0"])
            all_mu1.append(batch["mu1"])
    y0 = torch.cat(all_y0)
    y1 = torch.cat(all_y1)
    mu0 = torch.cat(all_mu0)
    mu1 = torch.cat(all_mu1)
    cov0_95, cov1_95, w0_95, w1_95 = coverage(y0, y1, mu0, mu1, level=0.95)
    cov0_99, cov1_99, w0_99, w1_99 = coverage(y0, y1, mu0, mu1, level=0.99)
    r0, r1 = rmse(y0, y1, mu0, mu1)
    return {
        "coverage_95_y0": cov0_95,
        "coverage_95_y1": cov1_95,
        "width_95_y0": w0_95,
        "width_95_y1": w1_95,
        "coverage_99_y0": cov0_99,
        "coverage_99_y1": cov1_99,
        "width_99_y0": w0_99,
        "width_99_y1": w1_99,
        "rmse_y0": r0,
        "rmse_y1": r1,
        "pehe": pehe(y0, y1, mu0, mu1),
    }


def _train_loop(
    model: _DiffusionBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    ckpt_path: str,
    log_fn: Callable | None = None,
    propnet: PropensityNet | None = None,
    early_stopping: bool = False,
) -> None:
    """MultiStepLR training with early stopping on total val ELBO.

    Saves checkpoint whenever val ELBO improves; patience armed after warmup_epochs.
    Loads best checkpoint into model before returning.
    log_fn:
        optional callable(log_dict: dict, step: int) -- called each epoch for wandb logging.
    """
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-6)
    p0, p1, p2, p3 = (int(f * cfg.train.epochs) for f in (0.25, 0.50, 0.75, 0.90))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )

    best_val_elbo = float("inf")
    patience_left = cfg.train.patience

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_losses: dict = defaultdict(float)
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            y = batch["y"].to(device)
            y_cf = batch["y_cf"].to(device)
            optimizer.zero_grad()
            comps = model.compute_loss(x, a, y, y_cf, propnet=propnet)
            loss = model.total_loss(comps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for k, v in comps.items():
                epoch_losses[k] += v.item()
            epoch_losses["total_loss"] += loss.item()
            n_batches += 1

        lr_scheduler.step()
        val_comps = calculate_val_loss(model, val_loader, device, propnet=propnet)

        if log_fn is not None:
            log = {f"train/{k}": v / n_batches for k, v in epoch_losses.items()}
            log.update({f"val/{k}": v for k, v in val_comps.items()})
            log_fn(log, epoch)

        val_loss = val_comps["total_loss"]
        log_msg = f"Epoch {epoch + 1}: val_elbo {val_loss:.4f}"

        if val_loss < best_val_elbo:
            best_val_elbo = val_loss
            torch.save(model.state_dict(), ckpt_path)
            if early_stopping and epoch >= cfg.train.warmup_epochs:
                patience_left = cfg.train.patience
            logger.info(f"{log_msg} ✓ (saved)")
        elif early_stopping and epoch >= cfg.train.warmup_epochs:
            patience_left -= 1
            logger.info(f"{log_msg} (patience {patience_left}/{cfg.train.patience})")
            if patience_left == 0:
                logger.info("Early stopping.")
                break
        else:
            logger.info(log_msg)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

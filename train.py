"""Shared training utilities: val_loss, evaluate, _train_loop."""

import csv
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.auxiliary import AuxOutcome
from src.config import Config
from src.metrics import coverage, pehe, rmse, wasserstein
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
            comps = model.compute_loss(x, a, y, y_cf, propnet)

            for k, v in comps.items():
                totals[k] += v.item()
            totals["total_loss"] += model.total_loss(comps).item()
            n += 1

    return {k: v / n for k, v in totals.items()}


def evaluate(
    model: _DiffusionBase,
    loader: DataLoader,
    K: int,
    device: torch.device,
    sigma: float,
    preds_csv_path: Path | None = None,
    clip_val: float | None = None,
) -> dict[str, float]:
    """Test-time evaluation: generate K PO samples and compute coverage, RMSE, PEHE, WD.

    sigma: true PO noise std (Hill's Setting B), in the same space as the model's
        outputs -- pass 1/y_std since outcomes are normalised (see src/metrics.wasserstein).
    preds_csv_path: if given, writes per-subject summary stats to a CSV for diagnostics.
    """
    model.eval()
    all_y0, all_y1, all_mu0, all_mu1 = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            y0_s, y1_s = model.sample_outcomes(x, a, K=K, clip_val=clip_val)  # each (B,K)
            all_y0.append(y0_s.cpu())
            all_y1.append(y1_s.cpu())
            all_mu0.append(batch["mu0"])
            all_mu1.append(batch["mu1"])

    y0 = torch.cat(all_y0)
    y1 = torch.cat(all_y1)
    mu0 = torch.cat(all_mu0)
    mu1 = torch.cat(all_mu1)

    if preds_csv_path is not None:
        lo0 = torch.quantile(y0, 0.025, dim=1)
        hi0 = torch.quantile(y0, 0.975, dim=1)
        lo1 = torch.quantile(y1, 0.025, dim=1)
        hi1 = torch.quantile(y1, 0.975, dim=1)

        preds_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(preds_csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # fmt: off
            writer.writerow(
                [
                    "mu0", "mu1",
                    "y0_mean", "y1_mean",
                    "y0_std", "y1_std",
                    "y0_lo95", "y0_hi95",
                    "y1_lo95", "y1_hi95",
                ]
            )
            # fmt: on

            for i in range(y0.shape[0]):
                # fmt: off
                writer.writerow(
                    [
                        mu0[i].item(), mu1[i].item(),
                        y0[i].mean().item(), y1[i].mean().item(),
                        y0[i].std().item(), y1[i].std().item(),
                        lo0[i].item(), hi0[i].item(),
                        lo1[i].item(), hi1[i].item(),
                    ]
                )
                # fmt: on

    cov_95_y0, cov_95_y1, width_95_y0, width_95_y1 = coverage(y0, y1, mu0, mu1, level=0.95)
    cov_99_y0, cov_99_y1, width_99_y0, width_99_y1 = coverage(y0, y1, mu0, mu1, level=0.99)
    rmse_y0, rmse_y1 = rmse(y0, y1, mu0, mu1)
    wass_y0, wass_y1 = wasserstein(y0, y1, mu0, mu1, sigma=sigma)

    return {
        "wasserstein_y0": wass_y0,
        "wasserstein_y1": wass_y1,
        "coverage_95_y0": cov_95_y0,
        "coverage_95_y1": cov_95_y1,
        "width_95_y0": width_95_y0,
        "width_95_y1": width_95_y1,
        "coverage_99_y0": cov_99_y0,
        "coverage_99_y1": cov_99_y1,
        "width_99_y0": width_99_y0,
        "width_99_y1": width_99_y1,
        "rmse_y0": rmse_y0,
        "rmse_y1": rmse_y1,
        "pehe": pehe(y0, y1, mu0, mu1),
    }


def _train_loop(
    model: _DiffusionBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    run_id: str,
    log_fn: Callable | None = None,
    propnet: PropensityNet | None = None,
) -> None:
    """MultiStepLR training with early stopping on total val ELBO.

    log_fn:
        optional callable(log_dict: dict, step: int) -- called each epoch for wandb logging.
    """
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-6)
    p0, p1, p2, p3 = (int(f * cfg.train.epochs) for f in (0.25, 0.50, 0.75, 0.90))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )

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
            comps = model.compute_loss(x, a, y, y_cf, propnet)
            loss = model.total_loss(comps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in comps.items():
                epoch_losses[k] += v.item()
            epoch_losses["total_loss"] += loss.item()
            n_batches += 1

        lr_scheduler.step()
        val_comps = calculate_val_loss(model, val_loader, device, propnet)

        if log_fn is not None:
            log = {f"train/{k}": v / n_batches for k, v in epoch_losses.items()}
            log.update({f"val/{k}": v for k, v in val_comps.items()})
            log_fn(log, epoch + 1)

        val_loss = val_comps["total_loss"]
        logger.info(
            f"Epoch {epoch + 1}:"
            f" train_elbo {epoch_losses['total_loss'] / n_batches:.4f},"
            f" val_elbo {val_loss:.4f}"
        )

    torch.save(
        model.state_dict(), Path(cfg.train.checkpoint_dir) / f"final_model_{run_id}.pth"
    )


def train_aux_outcome(
    aux: AuxOutcome,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    log_fn: Callable | None = None,
    patience: int = 10,
    min_epochs: int = 200,
) -> None:
    """Train AuxOutcome standalone via factual-only NLL: -log_prob(x, a, y_fac).mean().

    Pre-trains AuxOutcome before it's handed to HybridModel's own training loop (where it
    continues training via log_ry, unfrozen) -- removes the cold-start problem the old
    consistency loss's warmup ramp existed to protect against, without needing a ramp.
    Mirrors _train_loop's optimizer/LR-schedule/clipping shape, plus early stopping on val
    loss matching PropensityNet.fit's pattern (patience=10, min_epochs=200 defaults, same
    as PropensityNet's patience/n_iter_min). No checkpointing here -- _train_loop saves the
    full HybridModel, aux_outcome included, once the main loop finishes.
    """
    aux.to(device)
    optimizer = Adam(aux.parameters(), lr=cfg.train.lr, weight_decay=1e-6)
    p0, p1, p2, p3 = (int(f * cfg.train.epochs) for f in (0.25, 0.50, 0.75, 0.90))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )

    val_loss_best = float("inf")
    patience_left = patience
    best_state = None

    for epoch in range(cfg.train.epochs):
        aux.train()
        train_loss_sum, n_batches = 0.0, 0
        for batch in train_loader:
            x, a, y = batch["x"].to(device), batch["a"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            loss = -aux.log_prob(x, a, y).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aux.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        lr_scheduler.step()

        aux.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, a, y = batch["x"].to(device), batch["a"].to(device), batch["y"].to(device)
                val_loss_sum += (-aux.log_prob(x, a, y).mean()).item()
                n_val_batches += 1

        train_loss, val_loss = train_loss_sum / n_batches, val_loss_sum / n_val_batches
        if log_fn is not None:
            log_fn(
                {
                    "pretrain_aux/train_nll": train_loss,
                    "pretrain_aux/val_nll": val_loss,
                },
                epoch + 1,
            )
        logger.info(
            "[AuxOutcome pretrain] Epoch %d: train_nll %.4f, val_nll %.4f",
            epoch + 1, train_loss, val_loss,
        )

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            patience_left = patience
            best_state = {k: v.clone() for k, v in aux.state_dict().items()}
        else:
            patience_left -= 1

        if patience_left <= 0 and epoch >= min_epochs:
            logger.info("[AuxOutcome pretrain] Early stopping at epoch %d", epoch + 1)
            break

    if best_state is not None:
        aux.load_state_dict(best_state)
    aux.eval()

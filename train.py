"""Shared training utilities: val_loss, evaluate, _train_loop."""

import csv
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.auxiliary import AuxOutcome
from src.config import Config
from src.metrics import coverage, pehe, rmse, wasserstein
from src.model import HybridModel, _DiffusionBase
from src.propensity import PropensityNet
from src.zspace_ipw import calibration_diagnostic, effective_sample_size, zspace_ipw_weight

logger = logging.getLogger(__name__)

# Index of a_decoder's TTUR param group within _train_loop's optimizer, once
# ipw_model is not None. Hoisted to module level (rather than assigned inside the
# `if ipw_model is not None:` block that builds the optimizer) so basedpyright's
# definite-assignment analysis doesn't flag it as possibly-unbound at its use site
# several `if` blocks later, which it cannot correlate with that first block's guard.
_A_DECODER_GROUP = 1


def calculate_val_loss(
    model: _DiffusionBase,
    loader: DataLoader,
    device: torch.device,
    propnet: PropensityNet | None = None,
    epoch: int = 0,
    ema_a_decoder: EMA | None = None,
    ema_encoder: EMA | None = None,
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
            comps = model.compute_loss(
                x, a, y, y_cf, propnet, epoch, ema_a_decoder, ema_encoder
            )

            for k, v in comps.items():
                totals[k] += v.item()
            totals["total_loss"] += model.total_loss(comps).item()
            n += 1

    return {k: v / n for k, v in totals.items()}


def _log_ipw_diagnostics(
    model: HybridModel,
    train_loader: DataLoader,
    device: torch.device,
    ema_a_decoder: EMA,
    ema_encoder: EMA,
) -> dict[str, float]:
    """Aggregate pi_hat/a over the FULL training set and compute ESS & calibration.

    This isn't done per-batch as there would be too few subjects per calibration bin
    otherwise. No-grad, eval mode; purely observational, does not affect training.

    NOTE: the `ipw/*` row logged at step S reflects diagnostics computed from the EMA
    state as of the END of epoch S-1 (i.e. one epoch "behind" what `train/*` at the
    same step describes) -- `_train_loop` calls this after that epoch's EMA `.update()`
    calls but the values summarise the pass just completed. Also, the logged `ipw/ess`
    is computed from the un-ramped zspace_ipw_weight output below, not the actual
    ramp_weight-adjusted weights driving that epoch's loss -- so during ramp-up
    epochs, logged ESS is lower than the true in-loss ESS.
    """
    was_training = model.training
    model.eval()
    all_pi_hat, all_a = [], []
    with torch.no_grad():
        for batch in train_loader:
            x = batch["x"].to(device)
            a = batch["a"].to(device)
            y = batch["y"].to(device)
            pi_hat = model._compute_pi_hat(x, a, y, ema_encoder, ema_a_decoder)
            all_pi_hat.append(pi_hat)
            all_a.append(a)
    if was_training:
        model.train()

    pi_hat_all = torch.cat(all_pi_hat)
    a_all = torch.cat(all_a)
    # Read the clip threshold off the model itself (single source of truth) rather
    # than separately from cfg.diffusion.ipw_clip_prop -- the two happen to always
    # agree in practice but weren't enforced to.
    w = zspace_ipw_weight(pi_hat_all, a_all, model._ipw_clip_prop)
    ess = effective_sample_size(w)

    out = {"ess": ess, "ess_frac": ess / len(w)}
    out.update(calibration_diagnostic(pi_hat_all, a_all))
    return out


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
    ipw_model: HybridModel | None = (
        model if isinstance(model, HybridModel) and cfg.diffusion.use_ipw else None
    )

    # EMA for a_decoder and encoder, if using z-space IPW. Note that the EMA is only
    # used for inference in the loss computation, not for training the a_decoder itself.
    ema_a_decoder: EMA | None = None
    ema_encoder: EMA | None = None
    if ipw_model is not None:
        steps_per_epoch = len(train_loader)
        ema_kwargs = dict(
            beta=cfg.diffusion.ipw_ema_decay,
            min_value=cfg.diffusion.ipw_ema_decay,
            update_after_step=cfg.diffusion.ipw_ramp_start * steps_per_epoch,
            update_every=1,
        )
        ema_a_decoder = EMA(ipw_model.a_decoder, **ema_kwargs)
        ema_encoder = EMA(ipw_model.encoder, **ema_kwargs)

    # If using z-space IPW with TTUR, split the optimiser into two groups: one for the
    # a_decoder (group 1) and one for all other parameters (group 0). The a_decoder's
    # LR is multiplied by ttur_factor after ipw_ramp_start.
    use_two_lrs = ipw_model is not None and cfg.diffusion.ttur_factor != 1.0
    if use_two_lrs:
        a_decoder_param_ids = {id(p) for p in ipw_model.a_decoder.parameters()}
        other_params = [p for p in model.parameters() if id(p) not in a_decoder_param_ids]
        optimizer = Adam(
            [
                {"params": other_params, "lr": cfg.train.lr},
                {"params": list(ipw_model.a_decoder.parameters()), "lr": cfg.train.lr},
            ],
            weight_decay=1e-6,
        )
    else:
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
            loss_components = model.compute_loss(
                x, a, y, y_cf, propnet, epoch, ema_a_decoder, ema_encoder
            )
            loss = model.total_loss(loss_components)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if ipw_model is not None and ema_a_decoder is not None and ema_encoder is not None:
                ema_a_decoder.update()
                ema_encoder.update()

            for k, v in loss_components.items():
                epoch_losses[k] += v.item()
            epoch_losses["total_loss"] += loss.item()
            n_batches += 1

        lr_scheduler.step()

        if use_two_lrs and (epoch + 1) >= cfg.diffusion.ipw_ramp_start:
            # Re-derive from group 0's current (post-MultiStepLR-decay) LR each epoch
            # to avoid compounding the factor across epochs
            optimizer.param_groups[_A_DECODER_GROUP]["lr"] = (
                optimizer.param_groups[0]["lr"] * cfg.diffusion.ttur_factor
            )

        # ema_a_decoder/ema_encoder deliberately NOT passed here -- val/diffusion_loss
        # stays always-unweighted so it's a stable, directly-comparable monitoring
        # metric across use_ipw=True vs use_ipw=False runs
        val_loss_components = calculate_val_loss(model, val_loader, device, propnet, epoch)

        ipw_diagnostic: dict[str, float] = {}
        if (
            ipw_model is not None
            and ema_a_decoder is not None
            and ema_encoder is not None
            and (epoch + 1) >= cfg.diffusion.ipw_ramp_start
        ):
            ipw_diagnostic = _log_ipw_diagnostics(
                ipw_model, train_loader, device, ema_a_decoder, ema_encoder
            )

        if log_fn is not None:
            log = {f"train/{k}": v / n_batches for k, v in epoch_losses.items()}
            log.update({f"val/{k}": v for k, v in val_loss_components.items()})
            log.update({f"ipw/{k}": v for k, v in ipw_diagnostic.items()})
            log_fn(log, epoch + 1)

        val_total_loss = val_loss_components["total_loss"]
        logger.info(
            f"Epoch {epoch + 1}:"
            f" train_elbo {epoch_losses['total_loss'] / n_batches:.4f},"
            f" val_elbo {val_total_loss:.4f}"
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

    Pre-trains AuxOutcome before it's handed to HybridModel's own training loop.
    run_condition (experiment.py) freezes it immediately after this call returns
    -- freezing itself is the caller's responsibility, not this function's.

    Mirrors _train_loop's optimizer/LR-schedule/clipping shape, plus early stopping
    on val loss. No checkpointing here, as _train_loop saves the full HybridModel,
    aux_outcome included, once the main loop finishes.
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
    best_epoch = None

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
            "AuxOutcome pretrain epoch %d: train_nll=%.4f, val_nll=%.4f",
            epoch + 1,
            train_loss,
            val_loss,
        )

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            patience_left = patience
            best_state = {k: v.clone() for k, v in aux.state_dict().items()}
            best_epoch = epoch
        else:
            patience_left -= 1

        if patience_left <= 0 and epoch >= min_epochs:
            logger.info("AuxOutcome pretrain: Early stopping at epoch %d", epoch + 1)
            break

    if best_state is not None:
        aux.load_state_dict(best_state)
    aux.eval()

    if best_epoch is not None:
        logger.info("AuxOutcome pretrain: Restored best model from epoch %d", best_epoch + 1)

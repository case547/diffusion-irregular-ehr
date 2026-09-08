"""Reusable compute + statistics for the confounding-diffusion analysis notebook.

Pure functions -- no matplotlib. The notebook keeps the plotting.
"""

import numpy as np
import torch

from src.model import _DiffusionBase


def cross_evaluate(
    other_model: _DiffusionBase,
    y_traj: torch.Tensor,
    cond: torch.Tensor,
    a_cond: torch.Tensor,
) -> torch.Tensor:
    """`other_model`'s eps_theta at every (y_traj[i], tau=L-1-i) pair.

    The anchor trajectory's states `y_traj` are held fixed; only the model (and its
    conditioning `cond`) is swapped in. `cond` is what `other_model.encode_cond` returned
    for the anchor subjects -- x for DiffPO, the encoder posterior mean for HybridModel.

    y_traj: (L, N, 2). cond: (N, cond_dim). a_cond: (N,). Returns eps: (L, N, 2).
    """
    L, N, _ = y_traj.shape
    tau_grid = torch.arange(L - 1, -1, -1).unsqueeze(1).expand(L, N).reshape(-1)
    y_flat = y_traj.reshape(L * N, 2)
    cond_flat = cond.unsqueeze(0).expand(L, N, -1).reshape(L * N, -1)
    a_flat = a_cond.unsqueeze(0).expand(L, N).reshape(-1)
    with torch.no_grad():
        eps_flat = other_model.denoiser(y_flat, tau_grid, cond_flat, a_flat)
    return eps_flat.reshape(L, N, 2)


def divergence_ratio(
    div_norm: torch.Tensor, eps_norm: torch.Tensor, mask: torch.Tensor | slice
) -> tuple[float, float, float]:
    """mean ||d_tau||,  mean ||eps_theta||, and their ratio.

    This is computed over the (tau, subject) entries selected by mask.
    """
    div_norm_mean = div_norm[:, mask].mean()
    eps_norm_mean = eps_norm[:, mask].mean()
    return (
        div_norm_mean.item(),
        eps_norm_mean.item(),
        (div_norm_mean / eps_norm_mean).item(),
    )


def bootstrap_curve(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the per-tau mean of `values` (L, n) over its n subjects.

    Resamples subjects with replacement; returns (L, n_boot). The explicit loop keeps
    memory flat -- it never materialises the (L, n_boot, n) intermediate.
    """
    n = values.shape[1]
    return np.stack(
        [values[:, rng.integers(0, n, n)].mean(axis=1) for _ in range(n_boot)],
        axis=1,
    )


def pointwise_ci(boot: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) percentile band from a (L, n_boot) bootstrap array."""
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
    return lo, hi

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def wasserstein(
    y0: torch.Tensor,
    y1: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
) -> tuple[float, float]:
    """
    Mean per-subject 1-Wasserstein distance between K predictive samples and point mass at true PO.
    Returns (wd_y0, wd_y1).
    """
    y0_np = y0.cpu().numpy()
    y1_np = y1.cpu().numpy()
    mu0_np = mu0.cpu().numpy()
    mu1_np = mu1.cpu().numpy()
    wd0 = float(np.mean([wasserstein_distance(y0_np[i], [mu0_np[i]]) for i in range(len(mu0_np))]))
    wd1 = float(np.mean([wasserstein_distance(y1_np[i], [mu1_np[i]]) for i in range(len(mu1_np))]))
    return wd0, wd1


def coverage(
    y0: torch.Tensor,
    y1: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    level: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Predictive interval coverage and median width for each PO at the given confidence level.
    Returns (cov_y0, cov_y1, width_y0, width_y1).
    Coverage: fraction of subjects whose true PO falls inside [lo, hi].
    Width: median across subjects of hi[i] - lo[i]. Median is robust to outlier subjects
    where the model is very uncertain. Narrow + high coverage is the goal.
    """
    alpha = (1.0 - level) / 2.0
    lo0 = torch.quantile(y0, alpha, dim=1)
    hi0 = torch.quantile(y0, 1.0 - alpha, dim=1)
    lo1 = torch.quantile(y1, alpha, dim=1)
    hi1 = torch.quantile(y1, 1.0 - alpha, dim=1)
    cov0 = ((mu0 >= lo0) & (mu0 <= hi0)).float().mean().item()
    cov1 = ((mu1 >= lo1) & (mu1 <= hi1)).float().mean().item()
    width_y0 = (hi0 - lo0).median().item()
    width_y1 = (hi1 - lo1).median().item()
    return cov0, cov1, width_y0, width_y1


def rmse(
    y0: torch.Tensor,
    y1: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
) -> tuple[float, float]:
    """
    Per-PO RMSE using sample mean as point estimate.
    Returns (rmse_y0, rmse_y1).
    """
    r0 = (y0.mean(dim=1) - mu0).pow(2).mean().sqrt().item()
    r1 = (y1.mean(dim=1) - mu1).pow(2).mean().sqrt().item()
    return r0, r1


def pehe(
    y0: torch.Tensor,
    y1: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
) -> float:
    """
    √PEHE: root precision in estimating heterogeneous treatment effects.
    y0, y1: (B, K) PO samples; CATE estimated as y1.mean - y0.mean.
    mu0, mu1: (B,) true expected POs.
    """
    cate_est = y1.mean(dim=1) - y0.mean(dim=1)
    cate_true = mu1 - mu0
    return (cate_est - cate_true).pow(2).mean().sqrt().item()

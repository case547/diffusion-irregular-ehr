"""Pure functions for z-space IPW: trim/normalise weight, ramp, and observability.

Diagnostics (ESS, calibration).
"""

import torch


def zspace_ipw_weight(pi_hat: torch.Tensor, a: torch.Tensor, clip_prop: float) -> torch.Tensor:
    """Asymmetric arm-conditional trim, then normalise to mean 1.

    w = a/pi_hat + (1-a)/(1-pi_hat) only explodes as pi_hat->0 for treated subjects, and
    as pi_hat->1 for untreated subjects. Trimming is scoped to exactly those two cases,
    not a blanket band on pi_hat regardless of arm.

    Trimmed subjects fall back to raw weight 1 (not 0) so no training signal is lost; only
    the correction for that subject is declined.

    Shapes: `pi_hat`, `a`, and the return are all (B,).

    Assumes `pi_hat` is strictly within (0, 1). This relies on the caller's logit clamp
    (e.g. BernoulliNet's [-10, 10] clamp) to avoid NaN from a/pi_hat or (1-a)/(1-pi_hat)
    at the boundary.
    """
    raw_w = a / pi_hat + (1 - a) / (1 - pi_hat)
    overlap_ok = ((a == 1) & (pi_hat >= clip_prop)) | ((a == 0) & ((1 - pi_hat) >= clip_prop))
    w = torch.where(overlap_ok, raw_w, torch.ones_like(raw_w))
    return w / w.mean()


def ramp_weight(
    w: torch.Tensor, curr_epoch: int, ramp_start: int, ramp_end: int
) -> torch.Tensor:
    """Linearly interpolate w toward 1.0 between ramp_start and ramp_end (epochs).

    Mean-preserving for any ramp value as long as w itself already has mean 1 (true
    of `zspace_ipw_weight`'s output): E[1 + ramp*(w-1)] = 1 + ramp*(E[w]-1) = 1
    """
    ramp = min(1.0, max(0.0, (curr_epoch - ramp_start) / (ramp_end - ramp_start)))
    return 1.0 + ramp * (w - 1.0)


def effective_sample_size(w: torch.Tensor) -> float:
    """ESS = (sum w)^2 / sum(w^2).

    Equals len(w) when all weights are equal; collapses toward the count of a few
    dominant weights otherwise.
    """
    return (w.sum() ** 2 / (w**2).sum()).item()


def calibration_diagnostic(
    pi_hat: torch.Tensor, a: torch.Tensor, n_bins: int = 10
) -> dict[str, float]:
    """Bin subjects by predicted pi_hat (quantile bins), and compare each bin's mean
    prediction against its empirical treatment rate.

    `calib_mae` is the single scalar worth watching; per-bin values are for inspecting
    the reliability curve.
    """
    order = torch.argsort(pi_hat)
    bins = torch.tensor_split(order, n_bins)
    out: dict[str, float] = {}
    errs = []

    for i, idx in enumerate(bins):
        pred = pi_hat[idx].mean().item()
        empirical = a[idx].float().mean().item()

        out[f"calib_bin{i}_pred"] = pred
        out[f"calib_bin{i}_empirical"] = empirical
        errs.append(abs(pred - empirical))

    out["calib_mae"] = sum(errs) / len(errs)
    return out

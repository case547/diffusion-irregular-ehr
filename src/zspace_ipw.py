"""Pure functions for z-space IPW: trim/normalise weight, ramp, and observability.

Diagnostics (ESS, calibration).
"""

import torch


def zspace_ipw_weight(
    pi_hat: torch.Tensor,
    a: torch.Tensor,
    clip_prop: float,
    *,
    trim_to_zero: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Asymmetric arm-conditional trim, then (by default) normalise to mean 1.

    w = a/pi_hat + (1-a)/(1-pi_hat) only explodes as pi_hat->0 for treated subjects, and
    as pi_hat->1 for untreated subjects. Trimming is scoped to exactly those two cases,
    not a blanket band on pi_hat regardless of arm.

    trim_to_zero: if False (default), trimmed subjects fall back to raw weight 1, so no
        training signal is lost; only the correction for that subject is declined. If
        True, trimmed subjects fall back to 0 -- only safe when paired with a compensating
        signal for that subject elsewhere (e.g. the doubly-robust blend's plug-in term).
    normalize: if True (default), rescale to mean 1 across the batch. Set False for the
        doubly-robust blend, which needs the raw (unnormalised) weight for its AIPW-style
        correction -- renormalising would break the population-level unbiasedness property
        that makes the correction doubly robust in the first place.

    Shapes: `pi_hat`, `a`, and the return are all (B,).

    Assumes `pi_hat` is strictly within (0, 1). This relies on the caller's logit clamp
    (e.g. BernoulliNet's [-10, 10] clamp) to avoid NaN from a/pi_hat or (1-a)/(1-pi_hat)
    at the boundary.
    """
    raw_w = a / pi_hat + (1 - a) / (1 - pi_hat)
    overlap_ok = ((a == 1) & (pi_hat >= clip_prop)) | ((a == 0) & ((1 - pi_hat) >= clip_prop))
    fallback = torch.zeros_like(raw_w) if trim_to_zero else torch.ones_like(raw_w)
    w = torch.where(overlap_ok, raw_w, fallback)
    return w / w.mean() if normalize else w


def ramp_weight(
    w: torch.Tensor, curr_epoch: int, ramp_start: int, ramp_end: int
) -> torch.Tensor:
    """Linearly interpolate w toward 1.0 between ramp_start and ramp_end (epochs).

    Mean-preserving for any ramp value as long as w itself already has mean 1 (true of
    `zspace_ipw_weight`'s output when `normalize=True`, its default -- not true when
    called with `normalize=False`, e.g. for the doubly-robust blend):

        E[1 + ramp*(w-1)] = 1 + ramp*(E[w]-1) = 1

    The interpolation itself is correct either way: ramp=0 always yields w_eff=1 (no
    correction), ramp=1 always yields the full w.
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


def dr_blend_loss(
    w_eff: torch.Tensor, per_sample_real: torch.Tensor, per_sample_pseudo: torch.Tensor
) -> torch.Tensor:
    """Combine real and plug-in per-sample diffusion losses via a clamped AIPW-shaped blend.

    pseudo_coef = clamp(1 - w_eff, min=0), NOT the textbook unclamped (1 - w_eff): classical
    AIPW's negatively-weighted term multiplies a FIXED nuisance function, but here
    per_sample_pseudo is the SAME trainable model's own loss (its target is frozen, but its
    prediction is not) -- an unclamped negative coefficient lets gradient descent drive
    per_sample_pseudo, and hence the loss, to -inf with no bound. Clamping keeps both terms
    non-negative (each per_sample* is itself a sum of squared errors) while preserving the
    intended boundary case: w_eff=0 (fully trimmed) still gives pseudo_coef=1, full reliance
    on the plug-in term.

    For w_eff > 1 (well-overlapped subjects), pseudo_coef clamps to exactly 0 and this
    reduces to w_eff * per_sample_real alone -- identical to the single-term reweighting
    formula for those subjects; the two-term blend only genuinely activates for w_eff <= 1.

    Shapes: `w_eff`, `per_sample_real`, `per_sample_pseudo`, and the return are all (B,).
    """
    pseudo_coef = torch.clamp(1.0 - w_eff, min=0.0)
    return w_eff * per_sample_real + pseudo_coef * per_sample_pseudo

# Counterfactual Population-Mean Anchor for HybridModel

**Date:** 2026-08-31
**Status:** Draft

---

## Purpose

`HybridModel`'s diffusion loss only ever supervises the **factual** PO slot per subject
(`factual_mask` zeroes the counterfactual slot's gradient). Separately, `_noise_targets` still
constructs the counterfactual slot's *noised input* from the true `y_cf` — leak-free at high `tau`
in expectation but not in general, since `noisy_y`'s counterfactual entry is `sqrt(ab)*y_cf +
sqrt(1-ab)*eps` for every `tau`, including low `tau` where `sqrt(ab) -> 1`.

Two architecture-preserving interventions were tried earlier this project to address the resulting
Y0/Y1 calibration asymmetry: IPW reweighting of the diffusion loss via `PropensityNet`
(`use_propnet`), and an auxiliary counterfactual consistency loss using `AuxOutcome`'s cross-arm
prediction, masked to high `tau` for leakage reasons (`consistency_weight`). Both help; neither
removes the underlying leak, and the consistency loss is structurally barred from the low-`tau`
region where saturation has been shown to lock in.

This spec adds a third, independent mechanism: replace the counterfactual slot's true `y_cf` with a
**leak-free, non-oracle anchor** — the per-arm population mean from the training split — and replace
the hard `factual_mask` with a **soft mask** that gives the counterfactual slot a small, constant,
nonzero weight instead of zero. Because the anchor never depends on any individual subject's true
counterfactual value, this can safely apply across the *entire* `tau` schedule, including the
low-`tau` region the consistency loss couldn't reach.

A cheap empirical check (`arbitrary_evaluation.ipynb`) established that a trivial "always predict the
opposite arm's population mean" baseline — no model at all — substantially outperforms every real
`HybridModel` variant tried so far on PEHE (val, confounded: population-mean baseline 0.985 vs. best
real run 1.16). That result is the direct motivation for this design: anchoring the counterfactual
slot toward the same statistic the trivial baseline uses gives the model a floor to build from,
rather than leaving the counterfactual slot free to drift toward the clip boundary.

## Scope decisions (settled during brainstorming)

- **`HybridModel` only.** `DiffPO` is unchanged. Population means are a pure data statistic and
  could in principle apply to either model symmetrically, but the scope was deliberately kept to
  `HybridModel` for this iteration.
- **Runs alongside the existing consistency loss, not instead of it.** `consistency_weight` stays in
  the config and the codebase; whether a given run uses one, the other, both, or neither is a
  per-run config choice, not something this change forces.
- **Fixed constant weight, no ramp.** Unlike `AuxOutcome` (which starts near-random and needs a
  warmup while it converges), the population mean is exactly as good on step 1 as on the last step —
  there is no target-quality justification for ramping. A fixed weight from the start is also
  preferable for this mechanism's actual purpose: if clip-boundary saturation can start forming
  early in training, a ramped-in anchor would leave that window unprotected for no benefit.
- **Population means computed once, before training, from `train_ds`** (confounded or not, matching
  how `PropensityNet` is already fit on whatever was actually observed) — not recomputed per
  minibatch, which would be noisier for no benefit since the true population statistic is static.

## Design

### `src/config.py` — `DiffusionConfig`

One new field, following the `use_propnet`/`consistency_weight` off-by-default convention:

```python
cf_anchor_weight: float = 0.0  # soft-mask weight for the counterfactual slot, anchored to
                                # per-arm population means (0 = off, today's hard factual_mask)
```

### `experiment.py` — population-mean computation

A new helper, mirroring `_fit_propnet`'s shape and placement (computed once per replication, before
the training loop, from `train_ds`):

```python
def _compute_population_means(train_ds: CausalDataset) -> tuple[float, float]:
    """Per-arm mean factual outcome (normalised space) from the training split -- a leak-free
    anchor for HybridModel's counterfactual slot (see cf_anchor_weight)."""
    a0, a1 = train_ds.a == 0, train_ds.a == 1
    return train_ds.y[a0].mean().item(), train_ds.y[a1].mean().item()
```

Computed in the normalised space `train_ds.y` already lives in (the same space `_noise_targets`
operates in) — no rescaling needed, and consistent with how the trivial-baseline check itself was
computed.

### Threading

A new optional parameter, `pop_means: tuple[float, float] | None = None`, follows the exact path
`propnet` already takes: `run_condition` -> `_train_loop` / `calculate_val_loss` -> `model.compute_loss`.
`DiffPO.compute_loss` gains the same trailing parameter for interface compatibility (unused in its
body), matching how `epoch_frac` was added for the consistency loss. `_DiffusionBase.compute_loss`'s
abstract signature gains the same trailing parameter too, for the same reason `epoch_frac` was added
there rather than left implicit.

### `src/model.py` — `HybridModel.compute_loss`

No changes to `_noise_targets` or `calculate_diffusion_loss` — both already accept the relevant
values (`y_cf`, `factual_mask`) as plain tensors with no special semantics, so the substitution
happens entirely at the call site:

```python
if self._cf_anchor_weight > 0.0 and pop_means is not None:
    pm0, pm1 = pop_means
    # a=1 subjects: factual=y1, counterfactual=y0 -> anchor to pm0 (and vice versa)
    cf_target = torch.where(a == 1, torch.full_like(a, pm0), torch.full_like(a, pm1))
else:
    cf_target = y_cf

noisy_y, tau, eps, factual_mask = self._noise_targets(x.shape[0], x.device, a, y_fac, cf_target)
eps_pred = self.denoiser(noisy_y, tau, z, a)

if self._cf_anchor_weight > 0.0:
    soft_mask = factual_mask + self._cf_anchor_weight * (1.0 - factual_mask)
else:
    soft_mask = factual_mask

diffusion_loss = self.calculate_diffusion_loss(eps, eps_pred, soft_mask, x, a, propnet)
```

`HybridModel.__init__` gains `self._cf_anchor_weight = diffusion_cfg.cf_anchor_weight`, alongside the
existing `_consistency_weight` etc. assignments.

Gated the same way as `consistency_weight`: off by default (`cf_anchor_weight == 0.0`), and when off,
behavior is byte-for-byte identical to today (`cf_target is y_cf`, `soft_mask is factual_mask`).

## Testing

Mirror the consistency-loss test file's conventions in `tests/test_model.py`:

- **Inert by default:** with `cf_anchor_weight` at its default (0.0), `compute_loss`'s behavior
  (and specifically the diffusion loss's masking) is unchanged from today.
- **`cf_target` substitution:** with `cf_anchor_weight > 0.0` and known `pop_means`, assert that the
  *noised input*'s counterfactual slot is built from the population mean, not from `y_cf` — the
  test this mechanism exists for; a future refactor silently reintroducing the leak should fail this.
- **Soft mask arithmetic:** assert `soft_mask`'s counterfactual-slot entries equal
  `cf_anchor_weight` (not `0`) and factual-slot entries remain `1`, for a known `factual_mask`.
- **Finite loss:** `diffusion_loss` is finite with `cf_anchor_weight > 0.0`.

## Explicitly out of scope

- `DiffPO` (per the scope decision above).
- Any ramp/warmup for `cf_anchor_weight` (per the fixed-constant decision above).
- An `AuxOutcome`-based variant of this same mechanism (replacing the population mean with a
  per-subject `aux_outcome` estimate) — a plausible follow-up if this one shows promise, not
  attempted here. Known concern carried over from earlier discussion: `aux_outcome` conditions on
  raw `x`, not `z`, so training the counterfactual slot to chase its estimate risks capping quality
  at `aux_outcome`'s own level rather than improving on it — worth deciding deliberately if pursued,
  not folded into this change.
- Tuning `cf_anchor_weight`'s actual value for a real experiment run — this spec wires the mechanism
  with a sensible off-by-default config surface; picking a value that meaningfully affects the
  diagnosed asymmetry is a separate, later step.

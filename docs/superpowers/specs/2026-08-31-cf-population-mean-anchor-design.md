# Counterfactual Population-Mean Anchor

**Date:** 2026-08-31 (amended 2026-09-01: extended scope to `DiffPO`)
**Status:** Draft

---

**Amendment (2026-09-01):** the original version of this spec scoped the mechanism to
`HybridModel` only (see "Scope decisions" below, superseded). This revision extends it to
`DiffPO` as well. The base mechanism (config field, `experiment.py` population-mean computation,
`pop_means` threading through `run_condition`/`_train_loop`/`calculate_val_loss`) is unchanged and
already implemented on `feat/cf-population-mean-anchor`; this amendment only changes which model
classes the mechanism is wired into, and factors the previously `HybridModel`-only logic into a
shared `_DiffusionBase` helper so `DiffPO` doesn't duplicate it.

---

## Purpose

Both `HybridModel` and `DiffPO` share `_DiffusionBase`'s diffusion loss, which only ever supervises
the **factual** PO slot per subject (`factual_mask` zeroes the counterfactual slot's gradient) --
this is inherited machinery, not something specific to either model. Separately, `_noise_targets` still
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

- **Both `HybridModel` and `DiffPO` (revised 2026-09-01; originally `HybridModel` only).**
  Population means are a pure data statistic, computed identically regardless of which model
  consumes them, so there is no reason the mechanism can't apply to both. `DiffPO` gains its own
  `cf_anchor_weight`-gated behavior, independent of `HybridModel`'s — a run can anchor one, the
  other, both, or neither.
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
    anchor for HybridModel's and/or DiffPO's counterfactual slot (see cf_anchor_weight)."""
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

### `src/model.py` — shared `_DiffusionBase._apply_cf_anchor` helper

**Why a shared helper, not inline logic duplicated in both `compute_loss` methods:** the original
(`HybridModel`-only) version of this mechanism gated `cf_target`'s substitution and `soft_mask`'s
softening on two *separately written* `self._cf_anchor_weight > 0.0 and pop_means is not None`
checks — one before `_noise_targets`, one after. During implementation, those two checks drifted
out of sync (the `soft_mask` gate lost the `pop_means is not None` half), producing a broken partial
state: the mask softened while `cf_target` silently stayed `y_cf`. It was only caught by a real
end-to-end smoke test, not by unit tests (which always pass `pop_means` and `cf_anchor_weight`
together). Extending this to a second model (`DiffPO`) makes a second copy of that same
duplication-and-drift risk if the logic is written out twice more. A single shared helper, called
once per `compute_loss`, removes the risk structurally rather than relying on careful copying:

```python
def _apply_cf_anchor(
    self, a: torch.Tensor, y_cf: torch.Tensor, pop_means: tuple[float, float] | None
) -> tuple[torch.Tensor, bool]:
    """Leak-free counterfactual-slot substitution, shared by HybridModel and DiffPO.

    Returns (cf_target, anchor_active). cf_target replaces y_cf as _noise_targets' input
    when the anchor is active; anchor_active is the single source of truth the caller must
    reuse when deciding whether to also soften factual_mask once _noise_targets returns it
    -- see this spec's amendment for why that decision must not be re-derived twice.
    """
    anchor_active = self._cf_anchor_weight > 0.0 and pop_means is not None
    if not anchor_active:
        return y_cf, False
    pm0, pm1 = pop_means
    # a=1 subjects: factual=y1, counterfactual=y0 -> anchor to pm0 (and vice versa)
    cf_target = torch.where(a == 1, torch.full_like(a, pm0), torch.full_like(a, pm1))
    return cf_target, True
```

Both `HybridModel.__init__` and `DiffPO.__init__` gain
`self._cf_anchor_weight = diffusion_cfg.cf_anchor_weight` (alongside `HybridModel`'s existing
`_consistency_weight` etc. assignments; `DiffPO` gains only this one line, since it has no other
`diffusion_cfg`-derived instance attributes today).

### `HybridModel.compute_loss` (retrofit) and `DiffPO.compute_loss` (new)

Neither `_noise_targets` nor `calculate_diffusion_loss` change — both already accept the relevant
values (`y_cf`, `factual_mask`) as plain tensors with no special semantics. Both `compute_loss`
methods call the shared helper identically, differing only in what conditioning tensor
(`z` vs. `x`) reaches the denoiser -- exactly the same pre-existing difference that already exists
between the two methods today:

```python
cf_target, anchor_active = self._apply_cf_anchor(a, y_cf, pop_means)

noisy_y, tau, eps, factual_mask = self._noise_targets(x.shape[0], x.device, a, y_fac, cf_target)
eps_pred = self.denoiser(noisy_y, tau, z, a)  # DiffPO: self.denoiser(noisy_y, tau, x, a)

soft_mask = (
    factual_mask + self._cf_anchor_weight * (1.0 - factual_mask) if anchor_active else factual_mask
)
diffusion_loss = self.calculate_diffusion_loss(eps, eps_pred, soft_mask, x, a, propnet)
```

Gated the same way as `consistency_weight`: off by default (`cf_anchor_weight == 0.0`) on either
model independently, and when off, behavior is byte-for-byte identical to today for that model
(`cf_target is y_cf`, `soft_mask is factual_mask`).

### `experiment.py` — gating no longer keyed on `model_cls`

The `__main__` block currently computes `pop_means` only `if model_cls is HybridModel and
cfg.diffusion.cf_anchor_weight > 0.0`. Since `DiffPO` can now use the same mechanism, this becomes:

```python
pop_means = None
if cfg.diffusion.cf_anchor_weight > 0.0:
    pop_means = _compute_population_means(train_ds)
```

`_compute_population_means` itself is unchanged — it never depended on which model class was in
use, only on `train_ds`.

## Testing

`tests/test_model.py`'s existing `cf_anchor` tests (inert-by-default, `cf_target` substitution via
spying on `_noise_targets`, soft-mask arithmetic, finite loss) are unchanged — `HybridModel`'s
observable behavior doesn't change from this amendment, only its internals (calling the shared
helper instead of inlining the logic). Add the mirror set to `tests/test_diffpo.py`, following that
file's existing conventions (module-level `MODEL_CFG`/`DIFF_CFG`, a `_batch()` helper):

- **Inert by default:** with `cf_anchor_weight` at its default (0.0), `DiffPO.compute_loss`'s
  diffusion loss is unchanged from today.
- **`cf_target` substitution:** with `cf_anchor_weight > 0.0` and known `pop_means`, spy on
  `_noise_targets` the same way `tests/test_model.py` does, and assert the counterfactual slot's
  substitute value is the population mean, not `y_cf`.
- **Finite loss:** `diffusion_loss` is finite with `cf_anchor_weight > 0.0` on `DiffPO`.

A new shared test worth adding regardless of which model: assert `_apply_cf_anchor`'s
`anchor_active` return value is `False` whenever either `cf_anchor_weight == 0.0` or
`pop_means is None`, and `True` only when both conditions hold — this is the single test that
would have caught the original drift bug directly, rather than via an end-to-end smoke test.

## Explicitly out of scope

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

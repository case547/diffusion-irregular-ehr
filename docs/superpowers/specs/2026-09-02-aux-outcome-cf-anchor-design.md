# AuxOutcome-Anchored Counterfactual Slot for HybridModel

**Date:** 2026-09-02
**Status:** Draft

---

## Purpose

Two prior, separate mechanisms tried to give `HybridModel`'s counterfactual PO slot some
training signal (it otherwise gets zero direct gradient, since `factual_mask` only ever
supervises the factual slot):

1. **Auxiliary counterfactual consistency loss** (`consistency_weight`): an additive
   eps-space loss term using `aux_outcome.mean(x, 1-a)` as a pseudo-target, masked to
   high `tau` only (`consistency_min_tau_frac`) because `noisy_y`'s counterfactual slot
   is built from the true `y_cf` and leaks it back in at low `tau`, and ramped up over
   training (`consistency_warmup_frac`) because `aux_outcome` starts random and is
   trained jointly in the same loop.
2. **Counterfactual population-mean anchor** (`cf_anchor_weight`): substitutes the
   per-arm population mean for `y_cf` directly in `_noise_targets`' input (via
   `_apply_cf_anchor`), and softens `factual_mask` into a constant-weight `gradient_mask`.
   Leak-free by construction (the anchor never depends on any individual subject's `y_cf`),
   so it applies across the *entire* `tau` schedule — no masking needed. A cheap empirical
   check (`arbitrary_evaluation.ipynb`, further isolated to the val-only population by a
   follow-up script) established a population-mean-only baseline (no model at all) at
   PEHE≈0.985 (val, confounded) — a real real signal, but a per-arm *constant* with no
   `x`-conditioning, capped well below what a real regressor should achieve.

This spec **replaces both** with a single mechanism: anchor the counterfactual slot to a
**pre-trained `AuxOutcome`**'s per-subject prediction, reusing mechanism (2)'s proven
architecture (substitution + soft mask, safe across the full `tau` range) with a richer,
`x`-conditioned source than a per-arm constant. Pre-training removes mechanism (1)'s
cold-start problem without needing a ramp: if `aux_outcome` already has a good fit before
`HybridModel`'s own training starts, there's no "trusting a garbage teacher early" risk to
protect against.

**Scope: `HybridModel` only.** `DiffPO` has no `AuxOutcome` of its own and this spec does
not give it one — `DiffPO`'s prior use of the population-mean anchor (added in
`docs/superpowers/specs/2026-08-31-cf-population-mean-anchor-design.md`'s DiffPO
extension) is reverted in full, not left dormant.

## Design decisions (settled during brainstorming)

- **`HybridModel` only** — `_apply_cf_anchor` moves off the shared `_DiffusionBase` onto
  `HybridModel` itself (it referenced `self.aux_outcome`, which `DiffPO` doesn't have —
  keeping it shared was a latent `AttributeError` waiting on any config that set
  `cf_anchor_weight > 0` for `DiffPO`). `DiffPO.__init__`/`DiffPO.compute_loss` revert to
  their pre-cf-anchor form: no `_cf_anchor_weight`, no `_apply_cf_anchor` call.
- **Pre-train, don't freeze.** `aux_outcome` is trained standalone (via a new
  `train_aux_outcome`, ported from `auxoutcome_diagnostic.ipynb`/the standalone-diagnostic
  plan) *before* `HybridModel`'s main training loop starts, then continues training
  exactly as it does today — via its own `log_ry` term inside `compute_loss`, every epoch,
  never frozen. Pre-training only changes its *starting point*; the continuous-training
  behavior is unchanged from today.
- **Detachment kept.** `_apply_cf_anchor`'s use of `aux_outcome.mean(x, 1-a)` stays under
  `torch.no_grad()`, exactly as the old consistency loss did. `aux_outcome`'s parameters
  are updated only by its own `log_ry` objective — never by the diffusion loss via the
  anchor path. Without this, the two components could co-adapt into a mutually-reinforcing
  but not-necessarily-accurate state (`aux_outcome` drifting toward "whatever makes the
  denoiser's job easier" rather than genuinely modeling `y | x, a`).
- **Reuse the population-mean anchor's architecture, not the consistency loss's.**
  Substitute `cf_target` directly in `_noise_targets`' input (like mechanism (2)), rather
  than an additive eps-space term (mechanism (1)). This inherits "leak-free at every
  `tau`" for free — the true `y_cf` never enters the noised input at all, so unlike the
  consistency loss, no `min_tau` masking is needed.
- **Full removal, not default-off, for the consistency loss.** `consistency_weight`/
  `consistency_warmup_frac`/`consistency_min_tau_frac` config fields, the entire eps-space
  block in `HybridModel.compute_loss`, the `"consistency_loss" in components` guard in
  `total_loss`, and all of `tests/test_model.py`'s consistency-loss tests are deleted.
- **`epoch_frac` and `pop_means` are both fully removable**, not just simplified.
  `epoch_frac` had exactly one consumer in `compute_loss` (the consistency loss's ramp,
  confirmed via `grep` — no other reference in `src/model.py`'s `compute_loss` bodies);
  with that gone, it's dead in both `HybridModel.compute_loss` and `DiffPO.compute_loss`
  (interface-compatibility parameter), and in `_train_loop`/`calculate_val_loss`. `pop_means`
  is dead for the same reason mechanism (2) is being replaced — `_apply_cf_anchor`'s new
  form queries `self.aux_outcome` directly (already an owned attribute), so there's nothing
  left to thread in externally. Removing both simplifies `compute_loss`'s signature back
  down to `(x, a, y_fac, y_cf, propnet=None)` for both classes.
- **`cf_anchor_weight` keeps its name and config field** — the *mechanism* (substitution +
  soft mask) isn't changing, only the *source* of the anchor value. Recomment it to reflect
  the new semantics.

## Design

### `src/model.py`

**`_DiffusionBase`** — remove the abstract `compute_loss`'s `epoch_frac`/`pop_means`
trailing parameters, back to `(x, a, y_fac, y_cf, propnet=None)`. Remove `_apply_cf_anchor`
(moves to `HybridModel`) and its class-level `_cf_anchor_weight: float = 0.0` default (no
longer needed as a shared fallback once only one subclass uses it).

**`HybridModel`**:
- `__init__`: remove `_consistency_weight`/`_consistency_warmup_frac`/
  `_consistency_min_tau_frac` assignments. Keep `_cf_anchor_weight = diffusion_cfg.cf_anchor_weight`.
- New method, moved and adapted from the old shared `_apply_cf_anchor`:
  ```python
  def _apply_cf_anchor(
      self, x: torch.Tensor, a: torch.Tensor, y_cf: torch.Tensor
  ) -> tuple[torch.Tensor, bool]:
      """Leak-free counterfactual-slot substitution, anchored to a pre-trained (but still
      continuously training, per its own log_ry term) AuxOutcome's per-subject prediction.

      Returns (cf_target, anchor_active). Detached: gradient reaches only the denoiser,
      never aux_outcome, which is trained solely via its own log_ry term -- avoids the
      two components co-adapting into a mutually-reinforcing but inaccurate state.
      """
      anchor_active = self._cf_anchor_weight > 0.0
      if not anchor_active:
          return y_cf, False
      with torch.no_grad():
          cf_target = self.aux_outcome.mean(x, 1.0 - a)
      return cf_target, True
  ```
- `compute_loss`: signature drops `epoch_frac`/`pop_means`, back to `(x, a, y_fac, y_cf, propnet=None)`.
  Replace `cf_target, anchor_active = self._apply_cf_anchor(a, y_cf, pop_means)` with
  `cf_target, anchor_active = self._apply_cf_anchor(x, a, y_cf)` (only the call-site
  argument order/count changes — the surrounding `_noise_targets`/`gradient_mask`/
  `calculate_diffusion_loss` code is untouched, it was already written generically enough).
  Delete the entire `if self._consistency_weight > 0.0:` block and its two output keys.
- `total_loss`: delete the `if "consistency_loss" in components:` guard — back to the
  unconditional five-term sum.

**`DiffPO`**: revert `__init__` (no `_cf_anchor_weight` line) and `compute_loss` (no
`_apply_cf_anchor` call, no `gradient_mask` softening — back to `_noise_targets(..., y_cf)`
and `calculate_diffusion_loss(eps, eps_pred, factual_mask, x, a, propnet)` directly).
Signature drops `epoch_frac`/`pop_means`.

### `src/config.py`

Remove `consistency_weight`/`consistency_warmup_frac`/`consistency_min_tau_frac`. Keep and
recomment:
```python
cf_anchor_weight: float = 0.0  # soft-mask weight for the counterfactual slot, anchored to
                                # a pre-trained AuxOutcome's per-subject prediction
                                # (0 = off, today's hard factual_mask)
```

### `train.py`

**New `train_aux_outcome`** — ported from `auxoutcome_diagnostic.ipynb`/the standalone
plan (`/home/justin/.claude/plans/binary-dancing-haven.md`), with one adaptation:
checkpoint-saving is removed entirely, not made optional — `_train_loop` already
checkpoints the full `HybridModel` (including its now-further-trained `aux_outcome`) at
the end of the main loop, so this pre-training stage never needs to save anything of its
own; unlike the standalone diagnostic script, it never runs without that later save.

```python
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
            log_fn({"pretrain_aux/train_nll": train_loss, "pretrain_aux/val_nll": val_loss}, epoch + 1)
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
```

`calculate_val_loss`/`_train_loop`: drop `epoch_frac`/`pop_means` parameters and every
call site that passed them (`model.compute_loss(x, a, y, y_cf, propnet)`,
`calculate_val_loss(model, val_loader, device, propnet)`).

### `experiment.py`

Remove `_compute_population_means` entirely, and every place `pop_means` was threaded
(the `__main__` block's `pop_means = None / if cfg.diffusion.cf_anchor_weight > 0.0:`
block, and `run_condition`'s `pop_means` parameter and its pass-through to `_train_loop`).

Add, in `run_condition`, immediately after model construction and before `_train_loop`:
```python
if isinstance(model, HybridModel) and cfg.diffusion.cf_anchor_weight > 0.0:
    train_aux_outcome(model.aux_outcome, train_loader, val_loader, cfg, device)
```
Reuses the `train_loader`/`val_loader` `run_condition` already builds — no separate
DataLoaders needed. `min_epochs`/`patience` left at `train_aux_outcome`'s defaults (not
re-exposed as new config surface); revisit only if a real run shows they need tuning.

`from train import _train_loop, evaluate` gains `train_aux_outcome`.

## Testing

**`tests/test_model.py`**: delete the entire `# ── consistency loss ──` test block and
the existing `pop_means`-based `# ── cf population-mean anchor ──` tests (they test a
signature/mechanism that no longer exists). Replace with tests against the new
`_apply_cf_anchor(x, a, y_cf)`:
- **Inert by default**: `cf_anchor_weight` at its default (0.0) — `compute_loss`'s
  diffusion loss unchanged from the no-anchor path.
- **Substitution uses `aux_outcome`, not `y_cf`**: same spy-on-`_noise_targets` pattern as
  the old `test_cf_anchor_uses_population_mean_not_y_cf`, but asserting the captured
  `cf_target` matches `model.aux_outcome.mean(x, 1-a)` (computed independently in the
  test) rather than a `pop_means`-derived constant, and does not match the true `y_cf`.
- **Detachment**: same pattern as the old consistency loss's detachment test, simpler
  here since the anchor's effect is folded directly into `diffusion_loss` rather than a
  separate output key. With `cf_anchor_weight > 0`, call `.backward()` on
  `comps["diffusion_loss"]` alone (not the full `total_loss`, which also legitimately
  trains `aux_outcome` via `log_ry` and would confound the check): assert
  `model.denoiser.cond_proj.weight.grad is not None` and
  `model.aux_outcome.trunk[0].weight.grad is None`. If detachment were ever accidentally
  dropped, `cf_target`'s dependency on `aux_outcome.mean(x, 1-a)` would make this fail
  immediately, since `diffusion_loss`'s graph would then include `aux_outcome`'s parameters.
- **Soft mask arithmetic**: unchanged in spirit from the old
  `test_cf_anchor_soft_mask_weight`/`test_cf_anchor_softens_gradient_mask_in_compute_loss`
  — still valid since `gradient_mask`'s formula didn't change, only what feeds `cf_target`.

**`tests/test_diffpo.py`**: no new tests needed — confirms the revert by construction
(there's no `cf_anchor_weight`-related code path left in `DiffPO` for a test to exercise).
Existing tests should pass unmodified, confirming `DiffPO` is back to its pre-cf-anchor
behavior.

**`tests/test_training.py`**: add `train_aux_outcome` tests, following the standalone
diagnostic plan's spec minus checkpointing (no `checkpoint_path` parameter exists here to
test): loss decreases over a few epochs; early stopping actually stops early (not
dead code); best-state restored, not final-epoch state.

**`tests/test_experiment.py`**: update/add a test confirming `run_condition` calls
`train_aux_outcome` when `model_cls is HybridModel` and `cf_anchor_weight > 0`, and does
*not* call it for `DiffPO` or when `cf_anchor_weight == 0` (e.g. by monkeypatching
`train_aux_outcome` in the test module and asserting call/no-call).

## Explicitly out of scope

- Any ramp/warmup for `cf_anchor_weight` (unchanged decision from the population-mean
  anchor spec — pre-training removes the cold-start problem without one).
- Tuning `cf_anchor_weight`'s actual value, or `train_aux_outcome`'s `patience`/
  `min_epochs` defaults, for a real experiment run — this spec wires the mechanism;
  picking values that meaningfully affect the diagnosed asymmetry is a separate, later step.
- Extending this to `DiffPO` — explicitly reverted, not deferred; `DiffPO` would need its
  own `AuxOutcome` instance and pretraining step to ever support this, which is a
  materially bigger change than this spec's scope.
- Wiring `train_aux_outcome`'s `log_fn` to wandb inside `run_condition` — the parameter
  exists for interface consistency with `_train_loop`'s own `log_fn` convention, but
  `run_condition` can pass `None` for now; wiring it to the real wandb run is a small,
  separate follow-up if the pre-training curve turns out to be worth logging.

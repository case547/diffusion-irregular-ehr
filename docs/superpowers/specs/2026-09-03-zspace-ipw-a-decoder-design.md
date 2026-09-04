# Z-Space IPW for HybridModel via `a_decoder`

**Date:** 2026-09-03
**Status:** Proposed

---

## Purpose

The DiffPO-CEVAE hidden-confounders design doc (`2026-07-09-diffpo-cevae-hidden-confounders-design.md`) names two related, unresolved gaps in `HybridModel`:

- **Limitation 1, Auxiliary predictor bias:** `r_φ(y|x,a)` (`AuxOutcome`) is trained on observational data with no bias correction, propagating error into `z` and the diffusion model.
- **Limitation 2, No orthogonal loss:** DiffPO's Neyman-orthogonal, `π(x)`-based IPW reweighting isn't carried over to `HybridModel`, because unconfoundedness doesn't hold at the `x` level under hidden confounding — the correct propensity variable is `z`, not `x`. That doc's "Considered and Abandoned Approaches" section floats `π(z)`-based IPW as the theoretically correct fix, but rejects a naive version for three reasons: a circular dependency early in training (`z` shapes `π̂`, `π̂` reweights the loss that shapes `z`), `z` being a noisy sample rather than a fixed input (more variance than DiffPO's `π(x)`), and DiffPO's orthogonality proof not carrying over to a latent, learned conditioning variable.

This design builds a `z`-space IPW mechanism for `HybridModel`'s diffusion loss, using `a_decoder` (`p_ψ(a|z)`, already part of the model, trained every step via `log_pa`) as the propensity function — no new network. It is explicitly a mitigation of the *engineering* pathologies (circularity, noise) that made a naive version untenable, **not** a resolution of the missing orthogonality guarantee — see Limitations.

`use_ipw` (`src/config.py:21`, already present, currently a placeholder — `config/ihdp.yaml:17` comment: "something new is coming...") becomes the master gate for this mechanism.

---

## Method

### Composition

Eight sub-mechanisms compose into one weight computation, each targeting a specific failure mode identified during design:

| Mechanism | Targets |
|---|---|
| Ramp schedule | Circularity at initialization — no reweighting until `z`/`a_decoder` have had time to leave their random-init state |
| EMA (`a_decoder` + encoder) | The tightest form of circularity — this step's `z` reweighting this step's own loss |
| Multi-sample `z` | `z` being a noisy sample rather than a fixed covariate (DiffPO's `x` has no analogous noise) |
| Asymmetric trim → weight 1 | Unbounded weights from near-zero/near-one propensity, without discarding training signal on this small a dataset |
| Label smoothing on `a_decoder` | Overconfidence in `a_decoder` itself, reducing how often trimming fires and improving calibration in the untrimmed region |
| TTUR (`a_decoder`-only LR) | A second, independent lever on the same circularity EMA targets |
| ESS logging | Observability — are the resulting *weights* numerically healthy? |
| Calibration diagnostic | Observability — does `π̂(z)` track reality at all? |

The last two are complementary, not redundant: a batch can have healthy ESS (no single weight dominating) while `π̂` is still systematically miscalibrated, or vice versa. None of the eight targets the missing orthogonality proof (limitation 2's core gap) — see Limitations.

### 1. Ramp schedule

New fields `ipw_ramp_start`, `ipw_ramp_end` (epochs). Before `ipw_ramp_start`, the mechanism does not run at all — `w_eff = 1` for every subject, identical to today's behavior. Between the two, the weight linearly interpolates toward full strength; from `ipw_ramp_end` onward it's the full computed weight.

`ipw_ramp_start` is also the trigger point for two other things below: EMA buffer initialization, and the TTUR LR drop for `a_decoder`. All three start together — there's no separate "burn-in" concept.

### 2. EMA shadow weights

Built on the `ema_pytorch` library (already a dependency), rather than hand-rolled — two side-car `EMA` wrapper instances, one for `a_decoder`, one for the encoder, held as additional `HybridModel` attributes (`self._ema_a_decoder`, `self._ema_encoder`) alongside the live modules, **not** replacing `self.a_decoder`/`self.encoder` themselves (which stay exactly as-is so every existing call site is untouched):

```python
steps_per_epoch = len(train_loader)
ema_kwargs = dict(
    beta=cfg.diffusion.ipw_ema_decay,
    min_value=cfg.diffusion.ipw_ema_decay,   # collapses the library's own warmup ramp -- see below
    update_after_step=cfg.diffusion.ipw_ramp_start * steps_per_epoch,
    update_every=1,                          # library default of 10 is tuned for far longer runs
)
ema_a_decoder = EMA(model.a_decoder, **ema_kwargs)
ema_encoder = EMA(model.encoder, **ema_kwargs)
```

`EMA.update()` copies (not lerps) the live parameters into the shadow every call until `update_after_step`, then switches to exponential smoothing from there — so calling `.update()` unconditionally every training step (no separate epoch gate needed in our own code) reproduces exactly the "not maintained before `ipw_ramp_start`, initialized as an exact copy at that point" behavior established earlier in this design's discussion (continuously-copied-then-reset is mathematically identical to never-touched-until-init; the library just implements the former).

The library's own decay-warmup schedule (`inv_gamma`/`power`, ramping the effective decay up from 0 rather than jumping straight to `beta`) assumes runs of ≥10K–1M steps per its own documented presets — nothing close to this project's ~1500 total steps — so it's collapsed rather than tuned: passing `min_value=beta=ipw_ema_decay` forces the clamp in `get_current_decay()` to always evaluate to `ipw_ema_decay` immediately once past `update_after_step`, giving a plain fixed decay rate with none of the library's warmup ramp. `ipw_ema_decay` itself should still be chosen against this project's actual step count (~1500 total at `epochs=500`, `batch_size=256`, ~690 training subjects → ~3 batches/epoch) — a half-life in the tens of steps, not hundreds-to-thousands.

`ema_a_decoder.forward_eval(z)` / `ema_encoder.forward_eval(x, a, y_fac)` (no-grad, eval-mode, provided by the library) is used to compute the propensity estimate below (step 3). Never fed into the diffusion denoiser (which needs live, differentiable `z` to actually train the encoder — an EMA tensor carries no gradient history, and using it there would also introduce a train/inference mismatch, since inference always uses the live, final encoder). Never substituted into `a_decoder`'s own `log_pa` training either (step 7 below uses live `z`, live `a_decoder`).

### 3. Multi-sample `z` averaging

For `M = ipw_z_samples` draws, using the `EMA.forward_eval` helper from step 2 (already no-grad, eval-mode):

```python
probs = []
for _ in range(ipw_z_samples):
    mu, sigma = ema_encoder.forward_eval(x, a, y_fac)     # ZEncoder.forward, src/encoder.py:21-34
    z_m = mu + sigma * torch.randn_like(sigma)
    logits_m = ema_a_decoder.forward_eval(z_m)             # ADecoder.forward returns logits, src/decoders.py:33-36
    probs.append(torch.sigmoid(logits_m))
p_hat = torch.stack(probs).mean(dim=0)                 # average probabilities, not logits (Jensen)
```

`a_decoder.forward` returns raw logits (via `BernoulliNet.forward`, clamped to `[-10,10]`) — `sigmoid` is applied explicitly here since the `Bernoulli` distribution wrapper (which does this internally) isn't used; we only need the bare probability for the weight formula, not a distribution to sample or score.

Averaging is a Monte Carlo estimate of $\mathbb E_{z\sim q(z|x,a,y)}[p_\psi(a\mid z)]$, reducing the estimate's variance relative to a single stochastic `z` draw — this is the sub-mechanism targeting DiffPO's "noisier than `x`" objection specifically.

### 4. Asymmetric arm-conditional trim

$w = a/\hat\pi + (1-a)/(1-\hat\pi)$ only explodes as $\hat\pi\to 0$ for treated subjects, and as $\hat\pi\to 1$ for untreated subjects — not the reverse. Trimming is scoped to exactly those two dangerous cases, not a blanket band on $\hat\pi$ regardless of arm (which would needlessly discard already-well-behaved samples on the safe side of each arm):

```python
raw_w = a / p_hat + (1 - a) / (1 - p_hat)
overlap_ok = ((a == 1) & (p_hat >= ipw_clip_prop)) | ((a == 0) & ((1 - p_hat) >= ipw_clip_prop))
w = torch.where(overlap_ok, raw_w, torch.ones_like(raw_w))
```

`ipw_clip_prop` defaults to `0.1`, matching `gdr-learners`' `get_iptw` default (`../gdr-learners/src/models/utils.py:137-140`) rather than DiffPO's looser `0.05` `x`-space threshold (`DiffPO/src/main_model.py:144`) — deliberately, not by default inheritance. DiffPO's `0.05` is calibrated to a purpose-built, separately-validated classifier (`PropensityNet.fit`, its own train/val split and early stopping, fit on the fixed covariate `x`); `a_decoder` is trained jointly inside the larger ELBO on a noisy, still-learning latent `z`, and already needed multi-sampling, EMA, and label smoothing layered on top precisely because it's expected to be less reliable — matching DiffPO's looser bound would let that extra noise compound into larger surviving weights (`1/0.05=20` vs. `1/0.1=10`), not less. And `gdr-learners`' own `0.1` was chosen under a harsher trim-to-**zero** convention (a real signal-loss cost per trimmed subject); ours trims to weight 1 (step 4), a strictly cheaper consequence — if the tighter threshold was worth its full cost there, it's at least as justified here, where the same tightening only costs "no correction applied," not "no training signal." No separate magnitude clamp on top — a survivor's weight is automatically bounded by `1/ipw_clip_prop`, matching the `gdr-learners` `get_iptw` convention rather than DiffPO's belt-and-braces double clamp (prop-clamp *and* weight-clamp).

**Trimmed subjects get weight 1, not 0.** Two real-world precedents were checked in `../gdr-learners`: the doubly-robust two-stage estimator zeros trimmed weights but compensates with a plug-in pseudo-outcome loss term for the same subject (`../gdr-learners/src/models/two_stage_estimator.py:224-229`) — so no training signal is actually lost, just re-sourced; the plain "plugin"/one-stage IPTW estimator also zeros trimmed weights, with no such compensation (`../gdr-learners/src/models/plugins.py:176-178`), which is standard for a *pure* IPW estimator with no augmentation available. Neither precedent transfers cleanly: our diffusion loss is a single-term reweighted average like the plugin estimator (no compensating term), but `gdr-learners`' benchmarks run at roughly 100x more subjects than IHDP's ~690 training subjects — a trim event there is a rounding error; here it could plausibly zero out a meaningful fraction of an already-small batch. Weight 1 preserves full training signal (the model still needs to predict POs for every subject, trimmed or not) while declining to apply a correction that isn't trusted — a deliberate trade-off, not a free lunch (see Limitations).

### 5. Normalize, then ramp

```python
w = w / w.mean()
ramp = min(1.0, max(0.0, (epoch - ipw_ramp_start) / (ipw_ramp_end - ipw_ramp_start)))
w_eff = 1.0 + ramp * (w - 1.0)
```

Mean-preserving throughout the ramp automatically: since $\mathbb E[w]=1$ after normalizing, $\mathbb E[w_\text{eff}] = 1 + \text{ramp}\cdot(\mathbb E[w]-1) = 1$ for any ramp value — no additional renormalization needed after interpolating.

### 6. Applied to the diffusion loss

Same formula shape as today (`HybridModel.compute_loss`, `model.py:233-234`), weighted before the final mean:

```python
per_sample = (((eps_pred - eps) * gradient_mask) ** 2).sum(dim=1)   # (B,)
diffusion_loss = (per_sample * w_eff).mean()
```

### 7. `a_decoder`'s own training: label smoothing

`log_pa` (`model.py:218`, `self.a_decoder.log_prob(z, a).mean()`) keeps using **live** `z` (with grad, unaffected by EMA) — but against a smoothed target `a_smooth = a·(1-2ε) + ε` instead of raw `{0,1}`, `ε = a_decoder_label_smoothing`.

The Bernoulli log-likelihood $L(p)=t\log p+(1-t)\log(1-p)$ has $dL/dp=0$ exactly at $p=t$ — with $t=1-\varepsilon$ (a treated subject's smoothed target), the loss is *literally* minimized at $p=1-\varepsilon$, not at $p\to 1$. This gives `a_decoder` a genuine finite-loss equilibrium instead of an unreachable asymptotic pull toward the boundary (previously interrupted only by the raw-logit `±10` clamp, which itself permits `p` as extreme as `[0.00005, 0.99995]` — far looser than useful).

Given trimming (step 4) already hard-bounds the worst-case weight magnitude regardless of how extreme `p` gets, label smoothing's role here is secondary to trimming, not a replacement for it: fewer subjects near the trim boundary in the first place, and better-calibrated weights in the untrimmed middle region.

### 8. TTUR

`a_decoder`'s parameters get their own optimizer param group, separate from the rest of `HybridModel`. LR stays at the shared `cfg.train.lr` before `ipw_ramp_start`; drops to `cfg.train.lr * ttur_factor` from that epoch on — a discrete step (matching the existing `MultiStepLR` milestone pattern in `_train_loop`, `train.py:157-159`), not a smooth ramp. Targets the same circularity EMA does, via a different, independent lever (named after the analogous GAN-training technique, where it addresses the same class of co-adapting-networks problem).

### 9. Observability: ESS

$\text{ESS} = (\sum_i w_{\text{eff},i})^2 / \sum_i w_{\text{eff},i}^2$, logged (`ESS` and `ESS/B`) per epoch to wandb. Purely informational in this design — quantifies whether a batch's nominal size is being effectively eroded by a few large weights; does not feed back into any of the mechanisms above (percentile-based or ESS-driven adaptive clamping was considered and deferred — see below).

### 10. Observability: calibration diagnostic

Logged per epoch the way `PropensityNet.fit` already logs train/val loss (`src/propensity.py:179-187`, `logger.info` plus `log_fn`) — a second, independent diagnostic from ESS, checking whether `π̂(z)` tracks reality rather than whether the derived weights are numerically well-behaved. Bin subjects by predicted `p_hat` (quantile bins, so each bin is comparably populated regardless of how concentrated `p_hat` currently is — a fixed-width band could otherwise leave some bins nearly empty), and compare each bin's mean prediction against its empirical treatment rate:

```python
def calibration_diagnostic(p_hat: torch.Tensor, a: torch.Tensor, n_bins: int = 10) -> dict[str, float]:
    order = torch.argsort(p_hat)
    bins = torch.chunk(order, n_bins)
    out = {}
    errs = []
    for i, idx in enumerate(bins):
        pred, empirical = p_hat[idx].mean().item(), a[idx].float().mean().item()
        out[f"calib_bin{i}_pred"] = pred
        out[f"calib_bin{i}_empirical"] = empirical
        errs.append(abs(pred - empirical))
    out["calib_mae"] = sum(errs) / len(errs)
    return out
```

Computed once per epoch from `p_hat`/`a` aggregated across the full training set (not per-batch — at `batch_size=256` a single batch split ten ways leaves too few subjects per bin to read anything from; aggregated over ~690 training subjects each bin holds ~69). Reuses the same `p_hat` already computed in step 3 for that epoch's weighting, so this is close to free. `calib_mae` is the single scalar worth watching over training; the per-bin values are for occasionally inspecting the actual reliability curve, not for driving any decision automatically.

Together with ESS, this is what actually tells you whether burn-in was long enough, whether EMA is helping, or whether `ipw_clip_prop` needs adjusting — none of which is otherwise directly observable from downstream PEHE/RMSE.

---

## Architecture changes

- **`HybridModel.compute_loss`** needs a new `epoch: int` parameter (there is currently no epoch/step tracking passed into `compute_loss` at all in the present codebase — an earlier `epoch_frac`-based ramp mechanism for a different loss term existed in an older version of this file but is no longer present), threaded from `_train_loop`'s existing `for epoch in range(cfg.train.epochs)` loop (`train.py:161`) through both the training and validation call sites (`train.py:172`, and `calculate_val_loss`).
- **`ema_pytorch` (already a dependency)** supplies the EMA mechanics — two `EMA`-wrapped side-car instances (`a_decoder`, encoder), configured per step 2. No bespoke shadow-parameter helper needed.
- **`_train_loop`**: `a_decoder`'s parameters split into a separate `Adam` param group for the TTUR LR; `ema_a_decoder.update()`/`ema_encoder.update()` called once per training step (unconditionally — the library's own `update_after_step` handles the ramp-start gating internally, see step 2); ESS and the calibration diagnostic computed and logged once per epoch alongside existing epoch logging.
- **`src/config.py`**: seven new fields on `DiffusionConfig`.

## Config additions

```python
class DiffusionConfig(BaseModel):
    ...
    use_ipw: bool = False                     # already present — master gate for this mechanism
    ipw_ramp_start: int = 0                   # epoch; ramp begins, EMA init, TTUR LR drop
    ipw_ramp_end: int = 0                     # epoch; ramp reaches full weight
    ipw_clip_prop: float = 0.1                # asymmetric trim threshold
    ipw_z_samples: int = 5                    # M, MC-averaged propensity draws
    ipw_ema_decay: float = 0.0                # EMA decay for a_decoder + encoder shadow weights
    a_decoder_label_smoothing: float = 0.0    # epsilon; a_decoder's own log_pa target
    ttur_factor: float = 1.0                  # a_decoder LR multiplier from ipw_ramp_start onward
```

`use_ipw: False` short-circuits all nine steps above entirely — `compute_loss` falls straight back to today's unweighted `diffusion_loss` formula, and none of the other six fields are read. They only need valid (non-zero-division) values once `use_ipw: True`; their listed defaults are inert placeholders under the gate, not defaults meant to be used as-is with `use_ipw: True`.

---

## Considered and Abandoned Approaches

**Zero-weight trimming**, matching both `gdr-learners` precedents directly. Rejected in favor of weight-1 given IHDP's small training set relative to `gdr-learners`' benchmarks (see step 4) — this project's data scarcity, not a flaw in zero-weighting itself, which is legitimate, precedented practice for a plain reweighted-average loss.

**Doubly-robust blend**, using `aux_outcome`'s own prediction as a pseudo-outcome fallback specifically for trimmed subjects, mirroring `gdr-learners`' DR two-stage estimator and structurally identical to the model's existing `_apply_cf_anchor` mechanism. Considered, and reuses machinery already in the model — deferred in favor of plain multiplicative reweighting for this first iteration, revisit if weight-1 trimming proves insufficient in practice.

**Block-coordinate / EM-style alternation** (alternate N steps training encoder+denoiser with `a_decoder` genuinely frozen, M steps refitting `a_decoder` with the encoder frozen) as an alternative to continuous EMA smoothing. Cleaner decoupling within each phase, at the cost of transition instability and two new schedule hyperparameters. Parked.

**ESS-driven adaptive clamp bounds / percentile-based trimming**, replacing the fixed `ipw_clip_prop` threshold with one derived from the batch's own weight distribution or a tracked ESS target. Considered as the natural fuller version of ESS logging; deferred — ESS is observational only in this design, revisit if the fixed threshold doesn't track well in practice.

**Feeding EMA `z` into the diffusion denoiser itself**, not just into the propensity estimate. Rejected: EMA tensors carry no gradient history, so this would either block end-to-end encoder training through the diffusion loss entirely, or (if worked around) train the denoiser against a systematically lagged `z` — introducing a train/inference mismatch, since inference always uses the live, final encoder's `z`.

**Using `a_decoder` live (no EMA) for the propensity estimate.** Rejected: reintroduces the exact same-step circular dependency (this step's `z` reweighting this step's own loss) that EMA exists to break.

**Tightening the existing raw-logit clamp, or squashing the output range directly, as a substitute for label smoothing.** Considered; rejected as the *primary* mechanism against overconfidence, since it bounds only the achievable output range without changing the loss's minimizer (against hard `{0,1}` targets, the loss still has no interior minimum — gradients keep pushing toward the boundary, just saturating near it). Label smoothing gives a genuine equilibrium instead. Doubly moot as a primary defense once trimming (step 4) already hard-bounds weight magnitude regardless of `a_decoder`'s output range.

---

## Limitations

1. **Does not establish an orthogonality/robustness guarantee in the latent setting.** This design addresses the training-dynamics pathologies (circularity, sampling noise) that blocked a naive version — it does not derive, or claim, that the resulting reweighted diffusion loss retains any analogue of DiffPO's Neyman-orthogonality. That gap (limitation 2 in the parent design doc) remains open.
2. **Seven new hyperparameters**, several (`ipw_z_samples`, `ipw_ema_decay`, `ttur_factor`) with no principled way to set beyond empirical tuning at this project's specific scale (~690 training subjects, ~1500 total gradient steps at current config).
3. **EMA only smooths the propensity-estimate pathway.** `a_decoder`'s own training (`log_pa`, step 7) and the diffusion loss's `z` (step 6) both still use live values every step — this is a soft decoupling of one specific computation, not a hard freeze of anything.
4. **Weight-1 fallback silently forgoes correction exactly where it's needed most** — subjects with the most extreme apparent confounding are precisely the ones trimmed out of any bias adjustment. A deliberate trade-off favoring training-signal preservation on a data-scarce dataset, not a resolution of the underlying identification problem for those subjects.
5. **Fixed `ipw_clip_prop=0.1` may not generalize** across IHDP replications or `confounder_effect` settings without retuning — no adaptive mechanism is included in this design (see Considered and Abandoned).

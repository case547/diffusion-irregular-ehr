# DiffPO Confounding Diffusion Analysis

**Date:** 2026-08-18
**Status:** Draft

---

## Purpose

The [DiffPO-CEVAE design](2026-07-09-diffpo-cevae-hidden-confounders-design.md) motivates a latent
confounder $\mathbf{z}$ by arguing that DiffPO degrades under hidden confounding because it
conditions the diffusion denoiser directly on $\mathbf{x}$, which cannot represent an unobserved
confounder. That argument is currently only supported indirectly, via aggregate metrics
($\sqrt{\text{PEHE}}$, coverage, RMSE, Wasserstein distance) comparing DiffPO trained on clean vs. confounded IHDP.

This experiment inspects the denoising process itself — specifically the denoiser's predicted noise
$\boldsymbol{\epsilon}_\theta(y_\tau, \tau \mid \mathbf{x}, a)$ — to produce direct, mechanistic
evidence for three claims about plain DiffPO (no latent encoder, which will be a follow-up to this).

**Terminology.** $\boldsymbol{\epsilon}_\theta$ is what the network actually outputs and what is
computed, compared, and plotted everywhere below unless stated otherwise. It is related to, but not
the same as, the score function $\nabla_{y}\log p_\tau(y) = -\boldsymbol{\epsilon}_\theta(y,\tau) /
\sqrt{1-\bar\alpha_\tau}$ — the two are proportional at each fixed $\tau$ (opposite sign, scaled by
a $\tau$-dependent, position-independent factor), so a difference between two models' predicted
noise is also a difference between their score functions, but the magnitudes are not directly
comparable across $\tau$ without applying that scale factor. This spec says "predicted noise" or
"$\boldsymbol{\epsilon}_\theta$" throughout, and uses raw $\boldsymbol{\epsilon}_\theta$ — not the
rescaled score — for every quantity *except one*: the per-model field+density plot (2D vector-field
& density visualisation, item 1), which plots the score specifically so its arrows point toward the
density peaks it's paired with (a raw-$\boldsymbol{\epsilon}_\theta$ quiver there would visibly
point away from the peaks, since $\boldsymbol{\epsilon}_\theta$ is *subtracted* in the reverse
update — see that section for the full reasoning). Everywhere else — claims 1–3, and the standalone
difference field (item 2) — the operative quantity is raw $\boldsymbol{\epsilon}_\theta$, for the
reason below.

The rescaling is deliberately not used wherever two *models* are compared across $\tau$ (claim 2's
divergence-vs-$\tau$ plot, and item 2's difference field across its small-multiples snapshots). The
scale factor $1/\sqrt{1 - \bar\alpha_\tau} \to \infty$ as $\tau \to 0$, so rescaling would amplify
small $\tau \approx 0$ predicted-noise *disagreements* into large score disagreements — a plotting
artifact that could look like "divergence concentrates late" without the models actually disagreeing
more there. $\boldsymbol{\epsilon}_\theta$ has no such singularity: it is a direct regression target
toward unit Gaussian noise, well-behaved across the whole schedule (this is in fact why DDPM
parameterises networks to predict $\boldsymbol{\epsilon}_\theta$ rather than the score directly — a
better-conditioned training target). It is also the quantity that literally drives the DDIM/DDPM
update rule, so it is the more mechanistically direct thing to plot for those comparisons.

1. **A footprint exists.** Training DiffPO on confounded vs. clean IHDP produces a measurably
   different $\boldsymbol{\epsilon}_\theta$ function, not just a different aggregate accuracy
   number.
2. **The footprint has structure.** The disagreement between the two models' $\boldsymbol{\epsilon}_\theta$
   is not uniform across the diffusion timeline; where it concentrates (coarse/early $\tau \approx
   L$ vs. fine/late $\tau \approx 0$) says something about the nature of the confounding.
3. **The footprint predicts harm.** Subjects with larger $\boldsymbol{\epsilon}_\theta$ disagreement
   are the same subjects where DiffPO's point predictions are worse, and this concentrates on
   `momblack`-flipped subjects — tying the mechanism to the measured accuracy degradation.

This is scoped to `DiffPO` only (`src/model.py`) — no `DiffPOCEVAE`, no latent $\mathbf{z}$. The
sampler being added is written on the shared `_DiffusionBase` class, so it is available to
`DiffPOCEVAE` for a future extension of this analysis without rework.

---

## Method

### Data setup

Load the IHDP test split twice, from the same underlying subjects:

- **Clean:** `load_ihdp(...)` → `test_ds_clean`. Matches `naive_full` training condition.
- **Confounded:** `make_ihdp_confounded(test_ds_clean, effect=cfg.data.confounder_effect)` →
  `test_ds_conf`. Matches `naive_conf` training condition.

`make_ihdp_confounded` leaves $\mathbf{x}$ untouched and only flips $a$ (and correspondingly swaps
$y$/$y_\text{cf}$) for `momblack == 1` subjects. Consequently, for a `momblack == 0` subject,
`(x, a)` is *identical* between `test_ds_clean` and `test_ds_conf` — a built-in control group. For
`momblack == 1` subjects, the confounded model sees a flipped `a`.

Select a small, balanced subset of the test split for this first pass: ~10 `momblack == 1` +
~10 `momblack == 0` subjects. The two model checkpoints (`naive_full`, `naive_conf`) are supplied
by the user as notebook parameters, not hardcoded into this spec.

### DDIM sampler — `_ddim_reverse` on `_DiffusionBase`

Add a deterministic ($\eta = 0$) sibling to the existing `_ddpm_reverse` (`src/model.py`):

```python
def _ddim_reverse(
    self,
    BK: int,
    cond: torch.Tensor,
    a_rep: torch.Tensor,
    device: torch.device,
    y_init: torch.Tensor | None = None,
    clip_val: float | None = None,
    log_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

- `y_init`: if given, used as the starting noise $y_L$ instead of a fresh `torch.randn` draw. This
  is what guarantees two separate calls (one per model) start from the identical $z_T$ — passing
  an explicit tensor is more robust than reseeding the global RNG before each call (the pattern
  currently used ad hoc in `diffusion.ipynb`).
- Runs all `L` steps (`L=200` per `config/ihdp.yaml`) — no step-skipping. DDIM's usual benefit is
  sampling speed via a step subsequence; here the goal is temporal resolution to locate *where*
  divergence appears, so the full schedule is kept.
- Update rule (standard DDIM, $\eta=0$): at each step $\tau$, compute
  $\hat{y}_0 = (y_\tau - \sqrt{1-\bar\alpha_\tau}\,\epsilon_\theta) / \sqrt{\bar\alpha_\tau}$
  (optionally clipped, matching the existing `clip_val` convention), then
  $y_{\tau-1} = \sqrt{\bar\alpha_{\tau-1}}\,\hat{y}_0 + \sqrt{1-\bar\alpha_{\tau-1}}\,\epsilon_\theta$.
- `log_trajectory=True` additionally returns stacked `(L, BK, 2)` tensors of every intermediate
  $y_\tau$ and predicted $\epsilon_\theta(y_\tau, \tau)$.

`DiffPO` gets a thin public wrapper (parallel to `sample_outcomes`) exposing this for the analysis
notebook, e.g. `sample_ddim(x, a, y_init=None, log_trajectory=False)`.

The existing `_ddpm_reverse` also gains the same `log_trajectory` option (stochastic, no `y_init`
needed there since it isn't used for matched-seed comparisons) — used by the density-surface
visualisation below to collect a sample cloud of $\mathbf{y}_\tau$ at a chosen snapshot $\tau$.

### Two comparisons, not one

**(A) Independent trajectories.** Both models run their own `_ddim_reverse` from the same shared
$z_T$ (via `y_init`) per subject. Because $\epsilon_\theta^\text{clean}$ and
$\epsilon_\theta^\text{conf}$ generally disagree even at $\tau=L$, the two trajectories diverge
from step 1 onward and land on different final samples. This gives two independently-generated
endpoints per subject — used for **claim 3** (comparing point-estimate error against ground truth
$\mu_0, \mu_1$).

**(B) Cross-evaluated predicted-noise divergence.** Comparing $\epsilon_\theta$ predictions taken
from each model's *own* (already-diverged) trajectory conflates two effects — genuine disagreement
between the two models' $\epsilon_\theta$, and the fact that the two models are being queried at
different points entirely (drift). To isolate just the former, hold the query point fixed and swap
only the model:

$$\mathbf{d}_\tau = \boldsymbol{\epsilon}_\theta^\text{conf}(y_\tau^\text{clean}, \tau \mid x, a_\text{clean}) - \boldsymbol{\epsilon}_\theta^\text{clean}(y_\tau^\text{clean}, \tau \mid x, a_\text{clean})$$

$$\mathbf{d}'_\tau = \boldsymbol{\epsilon}_\theta^\text{conf}(y_\tau^\text{conf}, \tau \mid x, a_\text{conf}) - \boldsymbol{\epsilon}_\theta^\text{clean}(y_\tau^\text{conf}, \tau \mid x, a_\text{conf})$$

Every argument except which model's weights are used is held fixed within each formula — critically
including $a$: $\mathbf{d}_\tau$ uses $a_\text{clean}$ (the subject's `test_ds_clean` treatment) for
*both* terms, even the confound-model term, so for a `momblack == 1` subject the confound model is
deliberately queried with the treatment value it wasn't trained to expect for that subject. This is
intentional, not an oversight: using each model's own matched $a$ instead would reintroduce a
difference in the query point itself (this time in $a$ rather than $\mathbf{y}$), undoing the
purpose of the cross-evaluation construction. $\mathbf{d}_\tau$ anchors on the clean model's own
logged trajectory (`log_trajectory=True` from its `_ddim_reverse` run) and asks "what would the
confound model have predicted here?" — a single extra forward pass through
`confound_model.denoiser(y_tau, tau, x_rep, a_clean_rep)` per step, no new sampling loop.
$\mathbf{d}'_\tau$ is the mirror, anchored on the confound trajectory and using $a_\text{conf}$
throughout. Comparing the two: if they look similar, the disagreement is a property of the region
shared by both data manifolds; if they differ substantially, the disagreement itself depends
heavily on which model's states (and matching $a$) are used as the query point.

This is used for **claims 1–2**.

### Metrics and plots

- **Claim 1:** $\lVert \mathbf{d}_\tau \rVert$ and $\lVert \mathbf{d}'_\tau \rVert$, aggregated
  over all subjects and steps, are non-trivial relative to $\lVert \boldsymbol{\epsilon}_\theta
  \rVert$ itself.
- **Claim 2:** plot mean $\lVert \mathbf{d}_\tau \rVert$ (and $\mathbf{d}'_\tau$) vs. $\tau$
  ($L \to 0$), overall and split by `momblack`, to see whether divergence concentrates early
  (coarse/global) or late (fine-grained/local).
- **Claim 3:** separately, run each model's existing stochastic sampler (`sample_outcomes`, the
  current DDPM path, $K=50$ per `config/ihdp.yaml`) on the same subset to get point estimates;
  compute per-subject $|\hat{y} - \mu|$ against the ground-truth `mu0`/`mu1` already in the IHDP
  dataset. Compare per-subject trajectory-divergence (e.g. $\sum_\tau \lVert \mathbf{d}_\tau \rVert$)
  against per-subject error, stratified by `momblack`.

### 2D vector-field & density visualisation

Unlike typical diffusion applications (e.g. image pixels), the state space here is $\mathbf{y} =
[y_0, y_1] \in \mathbb{R}^2$ — the reverse process never leaves two dimensions. This means the
predicted-noise field $\boldsymbol{\epsilon}_\theta(\mathbf{y}, \tau)$, and the density surface it
implicitly shapes over the course of the reverse process, can both be visualised directly, with no
dimensionality reduction.

For each snapshot (a fixed subject's $(\mathbf{x}, a)$ and a timestep $\tau$), two plots are
produced:

1. **Per-model field + density, combined.** Generate a large batch ($K \approx 300$) of
   independent stochastic reverse trajectories for the subject via `_ddpm_reverse`'s
   `log_trajectory` option (see above), collect the sample cloud of
   $\mathbf{y}_\tau$ at the snapshot $\tau$, and fit a 2D KDE over that cloud to get
   $\hat{p}_\tau(\mathbf{y})$ on a grid — plotted as a `plot_surface`. This is sampling-based rather
   than reconstructing $\hat{p}_\tau$ by line-integrating $\boldsymbol{\epsilon}_\theta$ as if it
   were the score (see "Considered and Abandoned" for why — line integration requires the field to
   be curl-free, an unverified assumption; sampling+KDE needs no such assumption and reuses the same
   stochastic sampler already used for claim 3). On the same 3D axes, project the **score**
   $-\boldsymbol{\epsilon}_\theta(\mathbf{y}, \tau \mid \mathbf{x}, a) / \sqrt{1-\bar\alpha_\tau}$
   (evaluated on a matching 2D grid) as a 2D quiver on the floor of the axes (`zdir='z'`, offset
   below the surface's minimum). This is the one place in the spec that uses the rescaled score
   rather than raw $\boldsymbol{\epsilon}_\theta$: the score points toward increasing density by
   construction, so arrows correctly flow toward the peaks of the paired surface, whereas raw
   $\boldsymbol{\epsilon}_\theta$ points the opposite way (it is subtracted, not added, in the
   DDIM/DDPM update rule) and would visually contradict the surface it sits under. The
   $\tau\to0$ blow-up that rules out the score elsewhere (claim 1–2, item 2 below) doesn't apply
   here in the same way: those compare *disagreement between two models* across $\tau$, where the
   blow-up can manufacture an apparent difference out of noise; this plot shows one model's *own*
   field paired with its *own* density at a single $\tau$, where the score's steepening near
   $\tau\approx0$ reflects the real sharpening of the density toward the data manifold, not an
   artifact. Arrows drawn directly on the surface itself would be foreshortened and hard to read,
   so the floor projection keeps both readable in one figure. One such combined plot per model
   (clean, confound).
2. **Difference vector field, on its own plot.**
   $\mathbf{d}_\tau(\mathbf{y}) = \boldsymbol{\epsilon}_\theta^\text{conf}(\mathbf{y}, \tau \mid
   x, a) - \boldsymbol{\epsilon}_\theta^\text{clean}(\mathbf{y}, \tau \mid x, a)$, evaluated on the
   same grid, as a standalone 2D quiver plot — a direct visualisation of the "spurious vector field"
   the confounder introduces at that timestep. Kept separate rather than folded into (1), so the
   (typically much smaller) difference magnitude isn't visually swamped by the raw per-model field,
   and because it doesn't have a single natural density counterpart the way the per-model field
   does. The density difference surface $\Delta \hat{p}_\tau = \hat{p}_\tau^\text{conf} -
   \hat{p}_\tau^\text{clean}$ remains available as an optional additional plot alongside this one.

**Small multiples.** Both plot types are conditional on a fixed $(\mathbf{x}, a, \tau)$, so show a grid of
snapshots rather than one picture per plot type: a handful of representative timesteps (e.g.
$\tau \in \{L, 3L/4, L/2, L/4, 0\}$) across a couple of representative subjects (one
`momblack == 1`, one `momblack == 0`).

This complements claim 2's scalar $\lVert \mathbf{d}_\tau \rVert$-vs-$\tau$ plot: the scalar plot
shows *when* divergence peaks in aggregate, these views show *what shape* the divergence and the
underlying density have in outcome space at that moment, for a given subject.

### Deliverable

New notebook `confounding_diffusion.ipynb` (top-level, alongside `diffusion.ipynb` and
`analysis.ipynb`). Checkpoint paths for `naive_full`/`naive_conf` are parameters near the top of
the notebook. Figures saved to `images/`, following the convention set by `analysis.ipynb`.

---

## Considered and Abandoned Approaches

### Step-skipping DDIM for speed

DDIM's usual selling point is sampling in fewer steps than the trained schedule. Rejected here:
the analysis needs fine temporal resolution to locate *where* divergence appears (claim 2), and
`L=200` on a small subject subset is cheap enough that speed isn't a constraint.

### Notebook-local instrumented sampler (matching `diffusion.ipynb`'s existing pattern)

`diffusion.ipynb` already has a hand-rolled, notebook-local `ddpm_reverse` function used for the
clip-investigation analysis. Considered following that precedent again here. Rejected in favour of
a proper `_ddim_reverse` method on `_DiffusionBase`: this analysis needs the sampler called twice
per subject with careful control over shared initial noise and mid-trajectory model-swapping,
which is easier to get right (and reuse for `DiffPOCEVAE` later) as tested library code than as
duplicated notebook logic.

### Comparing raw per-model trajectories directly ($\epsilon_\theta^\text{conf}(y_\tau^\text{conf})$ vs. $\epsilon_\theta^\text{clean}(y_\tau^\text{clean})$)

This is the naive approach and was rejected because it conflates genuine disagreement between the
two models' $\epsilon_\theta$ with trajectory drift (see "Cross-evaluated predicted-noise
divergence" above) — a nonzero difference could mean the models disagree, or just that the two
states have drifted apart and any two unrelated $\epsilon_\theta$ evaluations differ. The
cross-evaluation construction removes this confound by holding the query point fixed.

### Reconstructing density by line-integrating $\boldsymbol{\epsilon}_\theta$ (as the score)

Rather than sampling+KDE, $\hat{p}_\tau(\mathbf{y})$ could be reconstructed by treating
$\boldsymbol{\epsilon}_\theta$ as (proportional to) the score and line-integrating it from a
reference point: $\log \hat{p}_\tau(\mathbf{y}) - \log \hat{p}_\tau(\mathbf{y}_\text{ref}) =
\int_{\mathbf{y}_\text{ref}}^{\mathbf{y}} \nabla\log p_\tau \cdot d\mathbf{y}$. This is only
path-independent (and thus well-defined) if the field is curl-free, which is not guaranteed for a
trained network and was not going to be verified for this first pass. Rejected in favour of
sampling+KDE, which needs no such assumption and reuses the stochastic sampler already required for
claim 3.

### Only computing $\mathbf{d}_\tau$ (one direction)

Initially scoped to anchor only on the clean trajectory. Extended to also compute $\mathbf{d}'_\tau$
(anchored on the confound trajectory) since the two give different, complementary information (see
above) at negligible extra cost — the confound trajectory is already being generated for claim 3's
independent-endpoint comparison.

---

## Limitations

1. **No training-variance null baseline.** This analysis uses a single `naive_full`/`naive_conf`
   checkpoint pair. A nonzero $\mathbf{d}_\tau$ could in principle partly reflect ordinary
   run-to-run training variance rather than the confounding mechanism specifically. A stronger
   version of claim 1 would compare against $\mathbf{d}_\tau$ computed between two independently
   trained *clean* checkpoints (multiple `naive_full` runs already exist in `checkpoints/`) as a
   noise floor. Left as a natural follow-up, not required for this first pass.

2. **Deterministic DDIM vs. stochastic DDPM evaluation.** Claims 1–2 use deterministic DDIM
   trajectories; claim 3's error metric uses the standard stochastic DDPM `sample_outcomes` path
   (matching how the models are actually evaluated elsewhere in the codebase). These are not the
   same sampling process, so the link drawn in claim 3 is correlational across subjects (higher
   divergence, higher error) rather than a proof that the DDIM-observed divergence directly causes
   the DDPM-sampled error.

3. **`momblack` proxy-recoverability caveat carries over.** As noted in the DiffPO-CEVAE design,
   `momblack` is not in `x1`–`x25` but may be partially recoverable via proxy covariates (site,
   maternal education). This means even `momblack == 0` subjects are not a perfectly clean control
   — DiffPO's $\boldsymbol{\epsilon}_\theta$ may already be indirectly influenced by the confounding
   mechanism through those proxies, which would attenuate (not eliminate) the clean/confound
   contrast this analysis is built to detect.

4. **Small subject subset.** ~20 subjects is enough to validate the pipeline and produce an
   illustrative figure, but not enough for a statistically rigorous claim about the general
   population. Scaling to the full test split (stratified by `momblack`) is a natural extension if
   the small-subset results look promising.

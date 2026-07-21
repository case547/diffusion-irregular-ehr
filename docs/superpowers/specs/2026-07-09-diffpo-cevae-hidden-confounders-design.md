# DiffPO-CEVAE: Extending DiffPO to Handle Hidden Confounders

**Date:** 2026-07-09
**Status:** Draft

---

## Purpose

DiffPO learns the full distribution of potential outcomes (POs) via a conditional diffusion model,
offering uncertainty quantification beyond point estimates. However, DiffPO assumes
**unconfoundedness** — that all confounders are observed in $\mathbf{x}$. This is rarely satisfied
in observational medical data, where unmeasured variables (e.g. disease severity, patient
behaviour) simultaneously influence treatment assignment and outcomes.

CEVAE addresses hidden confounding by introducing a latent variable $\mathbf{z}$ representing the
true confounder, but its outcome model is a simple Gaussian or Bernoulli — unable to capture
complex distributional structure.

This experiment proposes a hybrid: **CEVAE's latent confounder structure combined with DiffPO's
distributional diffusion outcome model**, yielding a method that handles hidden confounders while
learning the full PO distribution.

---

## Method

### Training Objective

The objective $\mathcal{F}$ is maximised jointly over parameters $(\phi, \psi, \theta)$:

$$
\mathcal{F} =
  \sum_{i=1}^{N} \left[
    \mathbb{E}_{\mathbf{z}_i \sim r_\phi(\mathbf{z}_i \mid \mathbf{x}_i, a_i, y_{i,0})}
    \left[ \log p_\psi(\mathbf{x}_i \mid \mathbf{z}_i) + \log p_\psi(a_i \mid \mathbf{z}_i) \right]
    - D_{\mathrm{KL}}\!\left(
        r_\phi(\mathbf{z}_i \mid \mathbf{x}_i, a_i, y_{i,0}) \,\|\, p(\mathbf{z}_i)
      \right)
  \right]
$$

$$
- \sum_{i=1}^{N} \mathbb{E}_{\substack{
    \mathbf{z}_i \sim r_\phi(\mathbf{z}_i \mid \mathbf{x}_i, a_i, y_{i,0}) \\
    \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
    \tau \sim \mathrm{Unif}\{1,\dots,L\}
  }}
  \left[ \left\| \epsilon - \epsilon_\theta(y_{i,\tau},\, \tau \mid \mathbf{z}_i, a_i) \right\|^2 \right]
+ \sum_{i=1}^{N} \left[
    \log r_\phi(a_i \mid \mathbf{x}_i) + \log r_\phi(y_{i,0} \mid \mathbf{x}_i, a_i)
  \right]
$$

where $y_{i,\tau} = \sqrt{\bar\alpha_\tau}\,y_{i,0} + \sqrt{1-\bar\alpha_\tau}\,\epsilon$ and
$p(\mathbf{z}_i) = \mathcal{N}(\mathbf{0}, \mathbf{I})$.

The three terms are:

1. **VAE reconstruction + regularisation (from CEVAE):** pushes $\mathbf{z}$ to encode the
   information in $(\mathbf{x}, a)$; the closed-form KL regularises toward the prior.
2. **Noise-matching diffusion loss (from DiffPO):** replaces CEVAE's simple $\log p(y_{i,0} \mid
   a_i, \mathbf{z}_i)$ Gaussian outcome model with a diffusion denoiser conditioned on $\mathbf{z}$
   rather than $\mathbf{x}$.
3. **Auxiliary prediction terms (from CEVAE):** trains $r_\phi(a \mid \mathbf{x})$ and
   $r_\phi(y_0 \mid \mathbf{x}, a)$ so that the encoder can be used at test time without observed
   $(a, y_0)$.

### Architecture

**Encoder** $r_\phi(\mathbf{z} \mid \mathbf{x}, a, y_0)$ — TARnet-split with shared base $g_1$
and treatment-specific heads $g_2, g_3$:

$$
(\boldsymbol{\mu}_{a=0}, \boldsymbol{\sigma}^2_{a=0}) = g_2(g_1(\mathbf{x}, y_0)),
\qquad
(\boldsymbol{\mu}_{a=1}, \boldsymbol{\sigma}^2_{a=1}) = g_3(g_1(\mathbf{x}, y_0))
$$

$$
\boldsymbol{\mu} = a\,\boldsymbol{\mu}_{a=1} + (1-a)\,\boldsymbol{\mu}_{a=0},
\qquad
\boldsymbol{\sigma}^2 = a\,\boldsymbol{\sigma}^2_{a=1} + (1-a)\,\boldsymbol{\sigma}^2_{a=0}
$$

At training, only the branch matching the observed $a_i$ is active and receives gradients;
each head specialises to its treatment group. $g_1$ is updated by all patients.

**Decoders** $p_\psi(\mathbf{x} \mid \mathbf{z})$, $p_\psi(a \mid \mathbf{z})$ — MLPs.
Training-only; discarded at inference.

**Diffusion denoiser** $\epsilon_\theta(y_\tau, \tau \mid \mathbf{z}, a)$ — U-Net with MLP
residual blocks (following DiffPO), conditioned on $\mathbf{z}$ rather than $\mathbf{x}$.
This is the key structural change from DiffPO.

**Auxiliary networks** $r_\phi(a \mid \mathbf{x})$, $r_\phi(y_0 \mid \mathbf{x}, a)$ — MLPs.
Trained jointly to supply missing inputs to the encoder at test time.

### Training Procedure

For each mini-batch $\{(\mathbf{x}_i, a_i, y_{i,0})\}_{i=1}^B$:

1. **Encode:** compute $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i)$ via TARnet-split encoder;
   sample $\hat{\mathbf{z}}_i = \boldsymbol{\mu}_i + \boldsymbol{\sigma}_i \odot \boldsymbol{\eta}_i$,
   $\boldsymbol{\eta}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (reparameterisation trick).
2. **VAE reconstruction:** compute $\log p_\psi(\mathbf{x}_i \mid \hat{\mathbf{z}}_i)$ and
   $\log p_\psi(a_i \mid \hat{\mathbf{z}}_i)$.
3. **KL regularisation:** compute in closed form
   $D_{\mathrm{KL}} = \frac{1}{2}\sum_j \left(\mu_{ij}^2 + \sigma_{ij}^2 - \log\sigma_{ij}^2 - 1\right)$.
4. **Diffusion forward:** sample $\tau \sim \mathrm{Unif}\{1,\dots,L\}$,
   $\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$; compute
   $y_{i,\tau} = \sqrt{\bar\alpha_\tau}\,y_{i,0} + \sqrt{1-\bar\alpha_\tau}\,\epsilon$.
5. **Noise prediction:** compute
   $\|\epsilon - \epsilon_\theta(y_{i,\tau}, \tau \mid \hat{\mathbf{z}}_i, a_i)\|^2$.
6. **Auxiliary:** compute $\log r_\phi(a_i \mid \mathbf{x}_i)$ and
   $\log r_\phi(y_{i,0} \mid \mathbf{x}_i, a_i)$.
7. **Update** $(\phi, \psi, \theta)$ by minimising $-\mathcal{F}$ via stochastic gradient descent.

### Inference Procedure

Given a patient $(\mathbf{x}^*, a^*)$ with observed covariates and treatment, produce $K$ samples
of each PO (matching DiffPO's convention of conditioning on observed $a$):

1. **Obtain $K$ patient-specific latent samples.**
   For $k = 1, \dots, K$:
   1. Sample $\hat{y}^{(k)} \sim r_\phi(y \mid \mathbf{x}^*,\, a^*)$ — impute the unobserved outcome via the auxiliary network.
   2. Sample $\hat{\mathbf{z}}^{(k)} \sim r_\phi(\mathbf{z} \mid \mathbf{x}^*,\, a^*,\, \hat{y}^{(k)})$.

2. **Reverse diffusion** for each $k = 1,\dots,K$:
   1. Sample $y_L^{(k)} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ — pure noise in $\mathbb{R}^2$ (both PO slots).
   2. For $\tau = L, L-1, \dots, 1$: compute
      $$\mu_\theta\!\left(y_\tau^{(k)}, \tau \mid \hat{\mathbf{z}}^{(k)}, a^*\right) = \frac{1}{\sqrt{\alpha_\tau}}\!\left(
        y_\tau^{(k)} - \frac{\beta_\tau}{\sqrt{1-\bar\alpha_\tau}}\,
        \epsilon_\theta\!\left(y_\tau^{(k)},\, \tau \mid \hat{\mathbf{z}}^{(k)},\, a^*\right)
      \right)$$
      then sample $y_{\tau-1}^{(k)} = \mu_\theta + \sigma_\tau \boldsymbol{\xi}$,
      $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$.

3. **Output:** the two slots of $y_0^{(k)}$ give samples from $p(Y(0) \mid \mathbf{x}^*, a^*)$
   and $p(Y(1) \mid \mathbf{x}^*, a^*)$ respectively, from which point estimates, predictive
   intervals, Wasserstein distances, and CATE estimates can be derived.

### Datasets and Experimental Structure

**Primary dataset: IHDP.** 747 patients, 25 features (6 continuous, 19 binary), binary treatment.
Real covariates and treatment assignments from a clinical trial; synthetic potential outcomes under
the standard "Response Surface B" setting. Used quantitatively by CEVAE and qualitatively by
DiffPO, making it the natural common ground for the hybrid.

**Secondary dataset: ACIC2018.** ~4,000 subjects, 177 anonymised covariates, binary treatment,
continuous outcomes. DiffPO's primary quantitative benchmark. Evaluating on ACIC2018 allows
direct comparison against DiffPO's published √PEHE numbers for the full-covariates condition and
tests whether the hybrid generalises beyond IHDP.

**Experimental structure: 2×2.** Method (DiffPO vs. DiffPO-CEVAE) crossed with data condition
(full covariates vs. hidden confounder):

- **Full covariates** — both methods trained on the standard covariates with original treatment
  assignments. Directly comparable to published CEVAE (IHDP) and DiffPO (ACIC2018) baselines.
  Establishes that DiffPO-CEVAE does not degrade when hidden confounding is absent; the latent
  z collapses toward the prior and the model approximately reduces to DiffPO.
- **Hidden confounder** — both methods trained on the same covariates with a treatment column
  corrupted by a hidden variable (see below). Tests the central hypothesis: DiffPO degrades,
  DiffPO-CEVAE partially recovers. Training covariates are identical across both conditions;
  only the treatment column differs.

**Introducing confounding on IHDP.** The `momblack` indicator (mother is Black) is obtained from
Hill's `sim.data` R object and is not present in the standard NPCI covariate set x1–x25. For
patients with momblack=1, treatment is flipped: $a_i \leftarrow 1 - a_i$. Factual outcomes
$y_{i,0}$ and ground-truth $\mu_0$, $\mu_1$ are left unchanged. momblack is not present in
x1–x25, so neither model observes it directly. However, race may be partially recoverable from
proxy variables already in x — site indicators and maternal education covariates correlate with
momblack — meaning DiffPO can indirectly account for some of the confounding through these
proxies. DiffPO-CEVAE's latent z provides a more principled mechanism for absorbing the residual
confounding that x cannot capture. The expected performance gap between methods is therefore
moderate rather than dramatic, which reflects realistic EHR settings where a hidden variable
typically has some observable correlates.

**Introducing confounding on ACIC2018.** With anonymised covariates there is no interpretable
demographic variable to withhold. A correlation-based mechanism is used instead: identify the
binary covariate most correlated with treatment assignment in each replicate, then flip treatment
for all subjects where that covariate equals 1. This is mathematically equivalent to the IHDP
mechanism and sufficient to test the hypothesis.

Both mechanisms are treatment label corruption rather than classical latent-variable confounding
(where z causally generates both a and y). They are realistic proxies for a demographic variable
absent from the EHR that systematically affected treatment allocation.

**Metrics:** √PEHE (primary causal accuracy), 95% predictive interval coverage (calibration),
Wasserstein-1 distance (distributional fit).

---

## Considered and Abandoned Approaches

### ACIC2018 as the sole benchmark dataset

Early design considered using ACIC2018 exclusively. This would allow citing DiffPO's published
√PEHE numbers directly for the full-covariates DiffPO condition and running DiffPO's existing
codebase on the confounded dataset without reimplementation. The dissertation framing as "DiffPO
with latent estimation" is also arguably more natural when evaluated on the dataset DiffPO was
designed for.

IHDP was added as the primary dataset because momblack provides a principled, interpretable
confounding mechanism with a real-world motivation, and CEVAE's published IHDP numbers provide
a natural reference for the full-covariates DiffPO-CEVAE condition. Both datasets are now used:
IHDP as the primary benchmark with an interpretable confounding narrative, ACIC2018 as a
secondary benchmark for direct comparison against DiffPO's published results.

### Simple concatenation encoder

Concatenating $(\mathbf{x}, a, y_0)$ into a single MLP ignores the structural difference between
treatment groups. The TARnet split is better motivated: treatment assignment is confounded with
$\mathbf{z}$, so the posterior over $\mathbf{z}$ may genuinely differ between treated and control
patients. Abandoned in favour of the TARnet-split encoder.

### Normalizing flow encoder

Replacing the diagonal Gaussian with a flow-based posterior allows richer, potentially multimodal
posteriors over $\mathbf{z}$. Abandoned for the initial experiment — the Gaussian approximation is
standard in CEVAE and sufficient to establish whether the hybrid works at all. A natural extension
if the Gaussian family proves too restrictive.

### Full marginalisation over $(a, y)$ at inference

Sampling $\hat{a} \sim r_\phi(a \mid \mathbf{x})$ and $\hat{y} \sim r_\phi(y \mid \mathbf{x}, \hat{a})$
to obtain $\hat{\mathbf{z}}$ avoids requiring the observed $a$ at test time. However, DiffPO
conditions on the observed $a$ at inference, and since the primary goal is close alignment with
DiffPO to isolate the effect of latent confounder estimation, we adopt the same convention.
Additionally, using the true $a$ is strictly more informative for encoding $\hat{\mathbf{z}}$
than marginalising over a propensity sample, and avoids a training/inference mismatch where the
denoiser sees a sampled $\hat{a}$ rather than the true $a$ it was trained with. The auxiliary
propensity network $r_\phi(a \mid \mathbf{x})$ is still trained (it contributes $\log r_\phi(a
\mid \mathbf{x})$ to the ELBO) but is not used at inference.

### Two-stage inference

Training a separate outcome model to produce $\hat{y}_0$ for the encoder (rather than training
$r_\phi(y_0 \mid \mathbf{x}, a)$ jointly within $\mathcal{F}$) introduces selection bias into
$\hat{y}_0$, which propagates into $\hat{\mathbf{z}}$. Joint training within the ELBO partially
corrects this through shared gradient pressure. Abandoned.

### IPW reweighting in $\mathbf{z}$-space

Replacing DiffPO's $\pi(\mathbf{x})$-based orthogonal loss with $\pi(\hat{\mathbf{z}})$-based
reweighting is theoretically motivated — unconfoundedness holds at the $\mathbf{z}$ level, not
the $\mathbf{x}$ level, so $\pi(\mathbf{z})$ is the correct propensity for IPW correction.
However, estimating propensity scores from noisy encoder samples $\hat{\mathbf{z}}$ creates a
circular dependency early in training (poor $\hat{\mathbf{z}}$ → poor $\hat\pi$ → poor
reweighting → poor $\hat{\mathbf{z}}$); $\hat{\mathbf{z}}$ is sampled, not observed, so
propensity estimates are noisier than in DiffPO; and DiffPO's Neyman-orthogonality proof does
not directly carry over to the latent setting. Left as a natural future extension.

---

## Limitations

1. **Auxiliary predictor bias.** The auxiliary outcome network $r_\phi(y_0 \mid \mathbf{x}, a)$
   is trained on observational data without IPW correction. Even with propensity-weighted
   marginalisation at inference, its counterfactual predictions remain biased, propagating error
   into $\hat{\mathbf{z}}$ and thus into the diffusion model.

2. **No orthogonal loss.** DiffPO's Neyman-orthogonal reweighting of the diffusion loss is not
   carried over, because unconfoundedness does not hold at the $\mathbf{x}$ level under hidden
   confounding. The causal adjustment relies entirely on the encoder recovering $\mathbf{z}$. If
   $\mathbf{z}$ is poorly estimated, confounding bias is not corrected and there is no
   first-order robustness guarantee.

3. **Non-identifiability of $\mathbf{z}$.** The latent confounder is not uniquely identified from
   observational data. Different $\mathbf{z}$ configurations may explain the observed
   $(\mathbf{x}, a, y_0)$ triples equally well. The model relies on the structural assumptions of
   the causal graph and the inductive bias of the encoder architecture to recover a useful
   $\mathbf{z}$.

4. **Unverifiable core assumption.** The central assumption $Y(a) \perp A \mid \mathbf{Z}$ cannot be
   tested from data. If the true causal graph differs — e.g. additional hidden confounders exist
   beyond those captured by $\mathbf{z}$ — the model produces biased PO estimates with no
   observable signal of failure.

5. **Gaussian variational family.** The diagonal Gaussian posterior may be too restrictive if the
   true posterior over $\mathbf{z}$ is multimodal or has strong inter-dimensional correlations.
   Misspecification of the variational family degrades the quality of inferred $\mathbf{z}$,
   propagating into the diffusion model.

6. **Causality or correlation.** The encoder is trained to maximise the ELBO — a generative
   objective that rewards explaining the observed joint distribution $p(\mathbf{x}, a, y_0)$.
   There is no causal supervision signal, so $\mathbf{z}$ may capture a statistical proxy that
   reproduces the correlations in the data without corresponding to the true hidden confounder.
   This is distinct from non-identifiability (limitation 3): even if $\mathbf{z}$ were uniquely
   determined by the data, it need not be causally interpretable.

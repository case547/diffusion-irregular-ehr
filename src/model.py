from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from src.auxiliary import AuxOutcome
from src.config import DiffusionConfig, ModelConfig
from src.decoders import ADecoder, XDecoder
from src.denoiser import Denoiser
from src.encoder import ZEncoder
from src.propensity import PropensityNet


class _DiffusionBase(nn.Module, ABC):
    """Shared noise schedule and DDPM helpers for HybridModel and DiffPO."""

    denoiser: Denoiser
    _cf_anchor_weight: float = 0.0

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
        propnet: PropensityNet | None = None,
        epoch_frac: float = 0.0,
        pop_means: tuple[float, float] | None = None,
    ) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def total_loss(self, components: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @abstractmethod
    def sample_outcomes(
        self, x: torch.Tensor, a: torch.Tensor, K: int = 50, clip_val: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s=0.008) -> np.ndarray:
        """Cosine schedule as proposed in Improved Denoising Diffusion Probabilistic Models.

        Nichol & Dariwhal, 2021 (https://arxiv.org/abs/2102.09672)
        """
        steps = timesteps + 1
        tau = np.linspace(0, timesteps, steps)
        alpha_bar = np.cos(((tau / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return np.clip(betas, 0.001, 0.999)

    def _init_schedule(self, d: DiffusionConfig) -> None:
        L = d.num_steps
        self.L = L

        if d.schedule == "cosine":
            betas = self.cosine_beta_schedule(L)
        elif d.beta_start is None or d.beta_end is None:
            raise ValueError(
                "beta_start and beta_end must be specified for linear or quad schedule"
            )
        elif d.schedule == "quad":
            betas = np.linspace(d.beta_start**0.5, d.beta_end**0.5, L) ** 2
        else:
            # Linear schedule
            betas = np.linspace(d.beta_start, d.beta_end, L)

        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)

        self.register_buffer("beta_sched", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alpha_sched", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alpha_bar_sched", torch.tensor(alpha_bar, dtype=torch.float32))

    @staticmethod
    def _assemble_yboth(
        a: torch.Tensor, y_fac: torch.Tensor, y_cf: torch.Tensor
    ) -> torch.Tensor:
        """Assemble [y0,y1] for each subject, given factual and counterfactual outcomes."""
        return torch.stack(
            [y_fac * (1 - a) + y_cf * a, y_fac * a + y_cf * (1 - a)], dim=1
        )  # (B,2)

    def _noise_targets(
        self,
        batch_size: int,
        device: torch.device,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble noised [y0,y1] and factual mask.

        Returns (noisy_y, tau, eps, factual_mask).
        """
        B = batch_size

        y_both = self._assemble_yboth(a, y_fac, y_cf)  # (B,2)
        factual_mask = torch.stack([1 - a, a], dim=1)  # (B,2)

        tau = torch.randint(0, self.L, (B,), device=device)
        eps = torch.randn(B, 2, device=device)
        ab_tau = self.alpha_bar_sched[tau].unsqueeze(1)  # (B,1)
        noisy_y = ab_tau.sqrt() * y_both + (1.0 - ab_tau).sqrt() * eps

        return noisy_y, tau, eps, factual_mask

    def calculate_diffusion_loss(
        self,
        eps: torch.Tensor,
        eps_pred: torch.Tensor,
        gradient_mask: torch.Tensor,
        x: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        propnet: PropensityNet | None = None,
    ) -> torch.Tensor:
        """Calculate the diffusion loss term E_z,τ,ε[‖ε - ε_θ(⋅)‖²]

        Optionally weighted by IPW if a PropensityNet is provided. Returns a scalar loss value.
        """
        per_sample = (((eps_pred - eps) * gradient_mask) ** 2).sum(dim=1)

        if propnet is not None:
            if x is None or a is None:
                raise ValueError(
                    "x and a must be provided if propnet is not None for IPW weighting"
                )

            with torch.no_grad():
                ipw = propnet.get_importance_weights(x, a)
            ipw = ipw.clamp(0.5, 3.0)
            ipw = ipw / ipw.mean()
            return (per_sample * ipw).mean()
        else:
            return per_sample.mean()

    def _apply_cf_anchor(
        self, a: torch.Tensor, y_cf: torch.Tensor, pop_means: tuple[float, float] | None
    ) -> tuple[torch.Tensor, bool]:
        """Leak-free counterfactual-slot substitution, shared by HybridModel and DiffPO.

        Returns `(cf_target, anchor_active)`.

        `cf_target` replaces `y_cf` as the input to `_noise_targets` when the anchor is
        active; `anchor_active` is the single source of truth the caller must reuse when
        deciding whether to also soften factual_mask once `_noise_targets` returns it.
        """
        anchor_active = self._cf_anchor_weight > 0.0 and pop_means is not None
        if not anchor_active:
            return y_cf, False

        pm0, pm1 = pop_means
        # For a=1 subjects: y_fac=Y(1), y_cf=Y(0) -> anchor Y(0) to pm0 (and vice versa)
        cf_target = torch.where(a == 1, torch.full_like(a, pm0), torch.full_like(a, pm1))
        return cf_target, True

    def _ddpm_reverse(
        self,
        BK: int,
        cond: torch.Tensor,
        a_rep: torch.Tensor,
        device: torch.device,
        clip_val: float | None = None,
    ) -> torch.Tensor:
        """DDPM reverse loop. cond is z (HybridModel) or x_rep (DiffPO). Returns (BK,2)."""
        y = torch.randn(BK, 2, device=device)

        for step in range(self.L - 1, -1, -1):
            tau = torch.full((BK,), step, device=device, dtype=torch.long)
            eps_pred = self.denoiser(y, tau, cond, a_rep)

            alpha_bar = self.alpha_bar_sched[step]
            beta = self.beta_sched[step]
            alpha = self.alpha_sched[step]

            mu = (1.0 / alpha.sqrt()) * (y - (beta / (1.0 - alpha_bar).sqrt()) * eps_pred)

            # If clipping is enabled, we compute a "clean" prediction of y and clip it to the
            # specified range. This is done to prevent extreme values in the reverse diffusion
            # process, which can lead to instability or unrealistic predictions (see notebooks)
            # `alpha_bar_safe` is used to avoid division by zero in the case of very small
            # alpha_bar values.
            if clip_val is not None:
                alpha_bar_safe = alpha_bar.clamp(min=1e-15)
                clean_pred: torch.Tensor = (
                    y - (1.0 - alpha_bar).sqrt() * eps_pred
                ) / alpha_bar_safe.sqrt()
                clean_pred = clean_pred.clamp(-clip_val, clip_val)

            if step > 0:
                alpha_bar_prev = self.alpha_bar_sched[step - 1]
                sigma = torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta)

                if clip_val is not None:
                    mu = (alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar)) * clean_pred + (
                        (1.0 - alpha_bar_prev) * alpha.sqrt() / (1.0 - alpha_bar)
                    ) * y

                y = mu + sigma * torch.randn_like(mu)
            else:
                y = mu if clip_val is None else clean_pred

        return y


class HybridModel(_DiffusionBase):
    """
    DiffPO-CEVAE: diffusion potential outcome model with latent hidden confounder.

    Objective (maximise):
      F = E_z[log p_ψ(x|z) + log p_ψ(a|z)]
          - KL[r_φ(z|x,a,y) ‖ N(0,I)]
          - E_z,τ,ε[‖ε - ε_θ(y_τ,τ|z,a)‖²]
          + log r_φ(y|x,a)

    Optional IPW weighting of the diffusion term via a pre-trained frozen PropensityNet,
    matching DiffPO's.
    """

    def __init__(self, model_cfg: ModelConfig, diffusion_cfg: DiffusionConfig):
        super().__init__()
        m = model_cfg
        self.encoder = ZEncoder(m.feature_dim, m.latent_dim, m.hidden_dim, m.num_layers)
        self.x_decoder = XDecoder(m.latent_dim, m.feature_dim, m.hidden_dim, m.num_layers)
        self.a_decoder = ADecoder(m.latent_dim, m.hidden_dim, m.num_layers)
        self.aux_outcome = AuxOutcome(m.feature_dim, m.hidden_dim, m.num_layers)
        self.denoiser = Denoiser(
            latent_dim=m.latent_dim,
            block_dim=diffusion_cfg.block_dim,
            hidden_dim=diffusion_cfg.hidden_dim,
            embedding_dim=diffusion_cfg.embedding_dim,
            num_blocks=diffusion_cfg.num_blocks,
            num_steps=diffusion_cfg.num_steps,
        )
        self._init_schedule(diffusion_cfg)
        self._consistency_weight = diffusion_cfg.consistency_weight
        self._consistency_warmup_frac = diffusion_cfg.consistency_warmup_frac
        self._consistency_min_tau_frac = diffusion_cfg.consistency_min_tau_frac
        self._cf_anchor_weight = diffusion_cfg.cf_anchor_weight

    def compute_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
        propnet: PropensityNet | None = None,
        epoch_frac: float = 0.0,
        pop_means: tuple[float, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Encode -- reparameterised; z retains grad for full end-to-end training
        z, mu, sigma = self.encoder.rsample(x, a, y_fac)

        log_px = self.x_decoder.log_prob(z, x).mean()
        log_pa = self.a_decoder.log_prob(z, a).mean()
        kl = 0.5 * (mu.pow(2) + sigma.pow(2) - 2.0 * sigma.log() - 1.0).sum(-1).mean()

        cf_target, anchor_active = self._apply_cf_anchor(a, y_cf, pop_means)

        noisy_y, tau, eps, factual_mask = self._noise_targets(
            x.shape[0], x.device, a, y_fac, cf_target
        )
        eps_pred: torch.Tensor = self.denoiser(noisy_y, tau, z, a)

        if anchor_active:
            gradient_mask = factual_mask + self._cf_anchor_weight * (1.0 - factual_mask)
        else:
            gradient_mask = factual_mask

        diffusion_loss = self.calculate_diffusion_loss(
            eps, eps_pred, gradient_mask, x, a, propnet
        )

        log_ry = self.aux_outcome.log_prob(x, a, y_fac).mean()

        out = {
            "log_px": log_px,
            "log_pa": log_pa,
            "kl": kl,
            "diffusion_loss": diffusion_loss,
            "log_ry": log_ry,
        }

        if self._consistency_weight > 0.0:
            # Pseudo-target for each subject's *counterfactual* arm, from the separately
            # trained aux_outcome regressor -- leak-free (aux_outcome only ever trains on
            # factual (x,a,y) triples). Detached: gradient must reach only the denoiser,
            # never aux_outcome (which is trained solely via its own log_ry term above).
            with torch.no_grad():
                y_pseudo_cf = self.aux_outcome.mean(x, 1.0 - a)

            # Place the pseudo value in whichever slot is counterfactual per subject
            pseudo_y_both = torch.stack([y_pseudo_cf * a, y_pseudo_cf * (1.0 - a)], dim=1)
            cf_mask = 1.0 - factual_mask

            # eps-space form: avoids dividing by sqrt(alpha_bar), equivalent to matching
            # the clean-data estimate to y_pseudo, but with no division by sqrt(alpha_bar)
            ab_tau = self.alpha_bar_sched[tau].unsqueeze(1)
            eps_target = (noisy_y - ab_tau.sqrt() * pseudo_y_both) / (
                1.0 - ab_tau
            ).sqrt().clamp(min=1e-6)
            cf_sq_err = (((eps_pred - eps_target) * cf_mask) ** 2).sum(dim=1)

            # Exclude low tau: ill-conditioned here (small divisor above), and where the
            # clean-data estimate would mostly leak the true (unobservable) y_cf baked
            # into noisy_y by _noise_targets, rather than test against y_pseudo.
            min_tau = int(self._consistency_min_tau_frac * self.L)
            tau_mask = (tau >= min_tau).float()

            consistency_raw = (cf_sq_err * tau_mask).sum() / tau_mask.sum().clamp(min=1.0)
            ramp = min(1.0, epoch_frac / max(self._consistency_warmup_frac, 1e-8))
            out["consistency_loss"] = (self._consistency_weight * ramp) * consistency_raw
            out["consistency_raw"] = consistency_raw.detach()

        return out

    def total_loss(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        """Minimise -F."""
        loss = (
            -components["log_px"]
            - components["log_pa"]
            + components["kl"]
            + components["diffusion_loss"]
            - components["log_ry"]
        )
        if "consistency_loss" in components:
            loss = loss + components["consistency_loss"]
        return loss

    @torch.no_grad()
    def sample_outcomes(
        self, x: torch.Tensor, a: torch.Tensor, K: int = 50, clip_val: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate K PO samples per subject. Returns y0 (B,K), y1 (B,K)."""
        B, device = x.shape[0], x.device
        BK = B * K
        x_rep = x.repeat_interleave(K, dim=0)
        a_rep = a.repeat_interleave(K, dim=0)
        y_hat = self.aux_outcome.sample(x_rep, a_rep)
        z, _, _ = self.encoder.rsample(x_rep, a_rep, y_hat)
        y = self._ddpm_reverse(BK, z, a_rep, device, clip_val).reshape(B, K, 2)
        return y[:, :, 0], y[:, :, 1]


class DiffPO(_DiffusionBase):
    """
    DiffPO baseline: diffusion PO model conditioned directly on x.

    Reimplemented using our stack for a fair comparison with HybridModel.
    Conditioning: [a, x] via Denoiser(latent_dim=feature_dim).
    Optional IPW weighting via a pre-trained frozen PropensityNet.
    """

    def __init__(self, model_cfg: ModelConfig, diffusion_cfg: DiffusionConfig):
        super().__init__()
        m = model_cfg
        self.denoiser = Denoiser(
            latent_dim=m.feature_dim,  # cond_proj takes [a, x]: size feature_dim+1
            block_dim=diffusion_cfg.block_dim,
            hidden_dim=diffusion_cfg.hidden_dim,
            embedding_dim=diffusion_cfg.embedding_dim,
            num_blocks=diffusion_cfg.num_blocks,
            num_steps=diffusion_cfg.num_steps,
        )
        self._init_schedule(diffusion_cfg)
        self._cf_anchor_weight = diffusion_cfg.cf_anchor_weight

    def compute_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
        propnet: PropensityNet | None = None,
        epoch_frac: float = 0.0,
        pop_means: tuple[float, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        cf_target, anchor_active = self._apply_cf_anchor(a, y_cf, pop_means)

        noisy_y, tau, eps, factual_mask = self._noise_targets(
            x.shape[0], x.device, a, y_fac, cf_target
        )
        eps_pred: torch.Tensor = self.denoiser(noisy_y, tau, x, a)

        if anchor_active:
            gradient_mask = factual_mask + self._cf_anchor_weight * (1.0 - factual_mask)
        else:
            gradient_mask = factual_mask

        diffusion_loss = self.calculate_diffusion_loss(
            eps, eps_pred, gradient_mask, x, a, propnet
        )
        return {"diffusion_loss": diffusion_loss}

    def total_loss(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        return components["diffusion_loss"]

    @torch.no_grad()
    def sample_outcomes(
        self, x: torch.Tensor, a: torch.Tensor, K: int = 50, clip_val: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate K PO samples per subject. Returns y0 (B,K), y1 (B,K)."""
        B, device = x.shape[0], x.device
        BK = B * K
        x_rep = x.repeat_interleave(K, dim=0)
        a_rep = a.repeat_interleave(K, dim=0)
        y = self._ddpm_reverse(BK, x_rep, a_rep, device, clip_val).reshape(B, K, 2)
        return y[:, :, 0], y[:, :, 1]

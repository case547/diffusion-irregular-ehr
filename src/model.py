from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from ema_pytorch import EMA

from src.auxiliary import AuxOutcome
from src.config import DiffusionConfig, VAEConfig
from src.decoders import ADecoder, XDecoder
from src.denoiser import Denoiser
from src.encoder import ZEncoder
from src.propensity import PropensityNet
from src.zspace_ipw import ramp_weight, zspace_ipw_weight


class _DiffusionBase(nn.Module, ABC):
    """Shared noise schedule and DDPM helpers for HybridModel and DiffPO."""

    denoiser: Denoiser

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
        propnet: PropensityNet | None = None,
        epoch: int = 0,
        ema_a_decoder: EMA | None = None,
        ema_encoder: EMA | None = None,
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

    No IPW weighting: unconfoundedness at x doesn't hold under hidden confounding, so an
    x-space PropensityNet would estimate the wrong propensity here (unlike DiffPO's).
    """

    def __init__(self, vae_cfg: VAEConfig, diffusion_cfg: DiffusionConfig):
        super().__init__()

        # Encoder-decoder stack for z-space latent confounder model
        vc = vae_cfg
        self.encoder = ZEncoder(
            vc.feature_dim, vc.latent_dim, vc.hidden_dim, vc.encoder_num_layers
        )
        self.x_decoder = XDecoder(
            vc.latent_dim, vc.feature_dim, vc.hidden_dim, vc.decoder_num_layers
        )
        self.a_decoder = ADecoder(
            vc.latent_dim, vc.a_decoder_hidden_dim, vc.decoder_num_layers
        )
        self.aux_outcome = AuxOutcome(vc.feature_dim, vc.hidden_dim, vc.aux_num_layers)

        self._a_decoder_label_smoothing = vae_cfg.a_decoder_label_smoothing

        # Diffusion denoiser for y-space potential outcome model, conditioned on z and a
        self.denoiser = Denoiser(
            input_dim=vc.latent_dim,
            block_dim=diffusion_cfg.block_dim,
            hidden_dim=diffusion_cfg.hidden_dim,
            embedding_dim=diffusion_cfg.embedding_dim,
            num_blocks=diffusion_cfg.num_blocks,
            num_steps=diffusion_cfg.num_steps,
        )
        self._init_schedule(diffusion_cfg)
        self._cf_anchor_weight = diffusion_cfg.cf_anchor_weight

        # IPW configuration
        # Field-range validation for the below lives on DiffusionConfig itself (see config.py)
        self._use_ipw = diffusion_cfg.use_ipw
        self._ipw_ramp_start = diffusion_cfg.ipw_ramp_start
        self._ipw_ramp_end = diffusion_cfg.ipw_ramp_end
        self._ipw_clip_prop = diffusion_cfg.ipw_clip_prop
        self._ipw_z_samples = diffusion_cfg.ipw_z_samples

    def _apply_cf_anchor(
        self, x: torch.Tensor, a: torch.Tensor, y_cf: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Leak-free counterfactual-slot substitution, anchored to a pre-trained
        AuxOutcome's per-subject prediction.

        Detached: gradient reaches only the denoiser, never aux_outcome, which is
        trained solely via its own log_ry term -- avoids the two components co-
        adapting into a mutually-reinforcing but inaccurate state.

        Returns (cf_target, anchor_active).
        """
        anchor_active = self._cf_anchor_weight > 0.0
        if not anchor_active:
            return y_cf, False
        with torch.no_grad():
            y_cf_pseudo = self.aux_outcome.mean(x, 1.0 - a)
        return y_cf_pseudo, True

    def _compute_pi_hat(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        ema_encoder: EMA,
        ema_a_decoder: EMA,
    ) -> torch.Tensor:
        """Multi-sample MC estimate of p_psi(a=1|z) via the EMA encoder + a_decoder.

        Takes `self._ipw_z_samples` independent `z` draws from q(z|x,a,y_fac) and averages
        `sigmoid(logits)` (not the logits themselves) using no_grad, eval-mode EMA models
        so this never affects (or is affected by) the current step's gradient.

        Returns (B,).
        """
        probs = []
        for _ in range(self._ipw_z_samples):
            mu, sigma = ema_encoder.forward_eval(x, a, y_fac)
            z_m = mu + sigma * torch.randn_like(sigma)
            logits_m = ema_a_decoder.forward_eval(z_m)
            probs.append(torch.sigmoid(logits_m))

        return torch.stack(probs).mean(dim=0)

    def compute_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
        propnet: PropensityNet | None = None,
        epoch: int = 0,
        ema_a_decoder: EMA | None = None,
        ema_encoder: EMA | None = None,
    ) -> dict[str, torch.Tensor]:
        # propnet accepted only for call-site parity with train.py's polymorphic
        # model.compute_loss(x, a, y, y_cf, propnet) -- never used: an x-space propensity
        # is the wrong nuisance function under hidden confounding (see class docstring).
        assert propnet is None, (
            "HybridModel does not support x-space IPW weighting -- see class docstring"
        )

        # Encode -- reparameterised; z retains grad for full end-to-end training
        z, mu, sigma = self.encoder.rsample(x, a, y_fac)

        # x decoding
        log_px = self.x_decoder.log_prob(z, x).mean()
        # a decoding, with label smoothing to avoid degenerate logit saturation
        eps_smooth = self._a_decoder_label_smoothing
        a_smooth = a * (1.0 - 2.0 * eps_smooth) + eps_smooth
        log_pa = self.a_decoder.log_prob(z, a_smooth).mean()
        # KL divergence term KL[r_φ(z|x,a,y) ‖ N(0,I)]
        kl = 0.5 * (mu.pow(2) + sigma.pow(2) - 2.0 * sigma.log() - 1.0).sum(-1).mean()

        cf_target, anchor_active = self._apply_cf_anchor(x, a, y_cf)

        noisy_y, tau, eps, factual_mask = self._noise_targets(
            x.shape[0], x.device, a, y_fac, cf_target
        )
        eps_pred: torch.Tensor = self.denoiser(noisy_y, tau, z, a)

        if anchor_active:
            gradient_mask = factual_mask + self._cf_anchor_weight * (1.0 - factual_mask)
        else:
            gradient_mask = factual_mask

        # Diffusion loss term E_z,τ,ε[‖ε - ε_θ(⋅)‖²]
        per_sample = (((eps_pred - eps) * gradient_mask) ** 2).sum(dim=1)

        if (
            self._use_ipw
            and ema_a_decoder is not None
            and ema_encoder is not None
            and epoch >= self._ipw_ramp_start
        ):
            pi_hat = self._compute_pi_hat(x, a, y_fac, ema_encoder, ema_a_decoder)
            w = zspace_ipw_weight(pi_hat, a, self._ipw_clip_prop)
            w_eff = ramp_weight(w, epoch, self._ipw_ramp_start, self._ipw_ramp_end)
            diffusion_loss = (per_sample * w_eff).mean()
        else:
            diffusion_loss = per_sample.mean()

        log_ry = self.aux_outcome.log_prob(x, a, y_fac).mean()

        out = {
            "log_px": log_px,
            "log_pa": log_pa,
            "kl": kl,
            "diffusion_loss": diffusion_loss,
            "log_ry": log_ry,
        }

        return out

    def total_loss(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        """Minimise -F."""
        return (
            -components["log_px"]
            - components["log_pa"]
            + components["kl"]
            + components["diffusion_loss"]
            - components["log_ry"]
        )

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

    def __init__(self, vae_cfg: VAEConfig, diffusion_cfg: DiffusionConfig):
        super().__init__()
        vc = vae_cfg
        self.denoiser = Denoiser(
            input_dim=vc.feature_dim,  # cond_proj takes [a, x]: size feature_dim+1
            block_dim=diffusion_cfg.block_dim,
            hidden_dim=diffusion_cfg.hidden_dim,
            embedding_dim=diffusion_cfg.embedding_dim,
            num_blocks=diffusion_cfg.num_blocks,
            num_steps=diffusion_cfg.num_steps,
        )
        self._init_schedule(diffusion_cfg)

    def compute_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        y_fac: torch.Tensor,
        y_cf: torch.Tensor,
        propnet: PropensityNet | None = None,
        epoch: int = 0,
        ema_a_decoder: EMA | None = None,
        ema_encoder: EMA | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the diffusion loss, optionally weighted by IPW from a `PropensityNet`,
        which should be frozen.

        `epoch`, `ema_a_decoder`, and `ema_encoder` are accepted only for call-site parity with
        `train.py`'s polymorphic `model.compute_loss(...)` -- `DiffPO` has no `a_decoder` or
        `encoder` to run z-space IPW against, so these are unused.
        """

        noisy_y, tau, eps, factual_mask = self._noise_targets(
            x.shape[0], x.device, a, y_fac, y_cf
        )
        eps_pred: torch.Tensor = self.denoiser(noisy_y, tau, x, a)

        per_sample = (((eps_pred - eps) * factual_mask) ** 2).sum(dim=1)

        if propnet is not None:
            with torch.no_grad():
                ipw = propnet.get_importance_weights(x, a)
            ipw = ipw.clamp(0.5, 3.0)
            ipw = ipw / ipw.mean()
            diffusion_loss = (per_sample * ipw).mean()
        else:
            diffusion_loss = per_sample.mean()

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

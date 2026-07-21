import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig, DiffusionConfig
from src.encoder import ZEncoder
from src.decoders import XDecoder, ADecoder
from src.auxiliary import AuxTreatment, AuxOutcome
from src.denoiser import Denoiser
from src.propensity import PropensityNet


class _DiffusionBase(nn.Module):
    """Shared noise schedule and DDPM helpers for DiffPOCEVAE and DiffPO."""

    def _init_schedule(self, d: DiffusionConfig) -> None:
        L = d.num_steps
        self.L = L
        betas = (
            (np.linspace(d.beta_start ** 0.5, d.beta_end ** 0.5, L)) ** 2
            if d.schedule == "quad"
            else np.linspace(d.beta_start, d.beta_end, L)
        )
        alpha = 1.0 - betas
        alpha_bar = np.cumprod(alpha)
        self.register_buffer("beta_sched", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alpha_sched", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("alpha_bar_sched", torch.tensor(alpha_bar, dtype=torch.float32))

    def _noise_targets(
        self, x: torch.Tensor, a: torch.Tensor, y_fac: torch.Tensor, y_cf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble noised [y0,y1] and factual mask. Returns (noisy_y, tau, eps, factual_mask)."""
        B = x.shape[0]
        y_both = torch.stack(
            [y_fac * (1 - a) + y_cf * a, y_fac * a + y_cf * (1 - a)], dim=1
        )  # (B,2)
        factual_mask = torch.stack([1 - a, a], dim=1)              # (B,2)
        tau = torch.randint(0, self.L, (B,), device=x.device)
        eps = torch.randn(B, 2, device=x.device)
        ab_tau = self.alpha_bar_sched[tau].unsqueeze(1)            # (B,1)
        noisy_y = ab_tau.sqrt() * y_both + (1.0 - ab_tau).sqrt() * eps
        return noisy_y, tau, eps, factual_mask

    def _ddpm_reverse(
        self, BK: int, cond: torch.Tensor, a_rep: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """DDPM reverse loop. cond is z (DiffPOCEVAE) or x_rep (DiffPO). Returns (BK,2)."""
        y = torch.randn(BK, 2, device=device)
        for step in range(self.L - 1, -1, -1):
            tau = torch.full((BK,), step, device=device, dtype=torch.long)
            eps_pred = self.denoiser(y, tau, cond, a_rep)
            alpha_bar = self.alpha_bar_sched[step]
            beta = self.beta_sched[step]
            alpha = self.alpha_sched[step]
            mu = (1.0 / alpha.sqrt()) * (y - (beta / (1.0 - alpha_bar).sqrt()) * eps_pred)
            if step > 0:
                alpha_bar_prev = self.alpha_bar_sched[step - 1]
                sigma = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta).sqrt()
                y = mu + sigma * torch.randn_like(mu)
            else:
                y = mu
        return y


class DiffPOCEVAE(_DiffusionBase):
    """
    DiffPO-CEVAE: diffusion potential outcome model with latent hidden confounder.

    Objective (maximise):
      F = E_z[log p_ψ(x|z) + log p_ψ(a|z)]
          - KL[r_φ(z|x,a,y) ‖ N(0,I)]
          - E_z,τ,ε[‖ε − ε_θ(y_τ,τ|z,a)‖²]
          + log r_φ(a|x) + log r_φ(y|x,a)
    """

    def __init__(self, model_cfg: ModelConfig, diffusion_cfg: DiffusionConfig):
        super().__init__()
        m = model_cfg
        self.encoder = ZEncoder(m.feature_dim, m.latent_dim, m.hidden_dim, m.num_layers)
        self.x_decoder = XDecoder(m.latent_dim, m.feature_dim, m.hidden_dim, m.num_layers)
        self.a_decoder = ADecoder(m.latent_dim, m.hidden_dim, m.num_layers)
        self.aux_treatment = AuxTreatment(m.feature_dim, m.hidden_dim, m.num_layers)
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

    def compute_loss(
        self, x: torch.Tensor, a: torch.Tensor, y_fac: torch.Tensor, y_cf: torch.Tensor,
        propnet=None,  # accepted for API compatibility with _train_loop; ignored
    ) -> dict[str, torch.Tensor]:
        # Encode -- reparameterised; z retains grad for full end-to-end training
        z, mu, sigma = self.encoder.rsample(x, a, y_fac)

        log_px = self.x_decoder.log_prob(z, x).mean()
        log_pa = self.a_decoder.log_prob(z, a).mean()
        kl = 0.5 * (mu.pow(2) + sigma.pow(2) - 2.0 * sigma.log() - 1.0).sum(-1).mean()

        noisy_y, tau, eps, factual_mask = self._noise_targets(x, a, y_fac, y_cf)
        eps_pred = self.denoiser(noisy_y, tau, z, a)
        diffusion_loss = (((eps_pred - eps) * factual_mask) ** 2).sum() / factual_mask.sum()

        log_qa = self.aux_treatment.log_prob(x, a).mean()
        log_qy = self.aux_outcome.log_prob(x, a, y_fac).mean()

        return {
            "log_px": log_px, "log_pa": log_pa, "kl": kl,
            "diffusion_loss": diffusion_loss, "log_qa": log_qa, "log_qy": log_qy,
        }

    def total_loss(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        """Minimise -F."""
        return (
            -components["log_px"] - components["log_pa"]
            + components["kl"] + components["diffusion_loss"]
            - components["log_qa"] - components["log_qy"]
        )

    @torch.no_grad()
    def sample_outcomes(
        self, x: torch.Tensor, a: torch.Tensor, K: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate K PO samples per subject. Returns y0 (B,K), y1 (B,K)."""
        B, device = x.shape[0], x.device
        BK = B * K
        x_rep = x.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)
        a_rep = a.unsqueeze(1).expand(B, K).reshape(BK)
        y_hat = self.aux_outcome.sample(x_rep, a_rep)
        z, _, _ = self.encoder.rsample(x_rep, a_rep, y_hat)
        y = self._ddpm_reverse(BK, z, a_rep, device).reshape(B, K, 2)
        return y[:, :, 0], y[:, :, 1]

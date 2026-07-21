import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Three-layer MLP: Linear(in,h) -> ReLU -> Linear(h,h) -> ReLU -> Linear(h,out)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding with two-layer MLP projection."""

    def __init__(self, num_steps: int, embedding_dim: int = 32):
        super().__init__()
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.proj1 = nn.Linear(embedding_dim, embedding_dim)
        self.proj2 = nn.Linear(embedding_dim, embedding_dim)

    def _build_embedding(self, num_steps: int, half_dim: int) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1).float()
        freqs = 10.0 ** (torch.arange(half_dim).float() / (half_dim - 1) * 4.0).unsqueeze(0)
        table = steps * freqs
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) long -> (B, embedding_dim)"""
        return F.silu(self.proj2(F.silu(self.proj1(self.embedding[tau]))))


class ResidualBlock(nn.Module):
    """DiffPO-style gated residual block.

    Adds diffusion time embedding, applies time_layer then feature_layer MLPs,
    gated activation (sigmoid*tanh), then splits into residual and skip connections.
    """

    def __init__(self, block_dim: int, embedding_dim: int):
        super().__init__()
        self.diffusion_projection = nn.Linear(embedding_dim, block_dim)
        self.time_layer = MLP(block_dim, 2 * block_dim, block_dim)
        self.feature_layer = MLP(block_dim, 2 * block_dim, block_dim)
        self.mid_projection = nn.Linear(block_dim, block_dim * 2)
        self.output_projection = nn.Linear(block_dim, block_dim * 2)

    def forward(
        self, x: torch.Tensor, diffusion_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_out, skip) both (B, block_dim)."""
        y = x + self.diffusion_projection(diffusion_emb)
        y = self.time_layer(y)
        y = self.feature_layer(y)
        gate, filt = self.mid_projection(y).chunk(2, dim=-1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.output_projection(y).chunk(2, dim=-1)
        return (x + residual) / math.sqrt(2.0), skip


class Denoiser(nn.Module):
    """ε_θ(noisy_y, τ | z, a) -- DiffPO-style denoiser conditioned on latent z and treatment a.

    diff_input = cond_proj([a, z]) + mapping_noise(noisy_y); ResidualBlocks refine;
    dual output heads produce joint [ε̂_y0, ε̂_y1] prediction.
    """

    def __init__(
        self,
        latent_dim: int,
        block_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_blocks: int,
        num_steps: int,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.cond_proj = nn.Linear(latent_dim + 1, block_dim)  # projects [a, z]
        self.mapping_noise = nn.Linear(2, block_dim)
        self.diffusion_embedding = DiffusionEmbedding(num_steps=num_steps, embedding_dim=embedding_dim)
        self.residual_layers = nn.ModuleList(
            [ResidualBlock(block_dim, embedding_dim) for _ in range(num_blocks)]
        )
        self.output_projection1 = nn.Linear(block_dim, hidden_dim)
        self.output_projection2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_projection2.weight)
        self.y0_layer = nn.Linear(hidden_dim, hidden_dim)
        self.y1_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_y0 = nn.Linear(hidden_dim, 1)
        self.output_y1 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        noisy_y: torch.Tensor,
        tau: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """noisy_y: (B,2), tau: (B,) long, z: (B, latent_dim), a: (B,) -> (B,2)"""
        cond = torch.cat([a.unsqueeze(-1), z], dim=-1)            # [B, latent_dim+1]
        h = F.relu(self.cond_proj(cond) + self.mapping_noise(noisy_y))  # [B, block_dim]
        diff_emb = self.diffusion_embedding(tau)
        skips = []
        for layer in self.residual_layers:
            h, skip = layer(h, diff_emb)
            skips.append(skip)
        h = torch.stack(skips).sum(dim=0) / math.sqrt(self.num_blocks)  # [B, block_dim]
        h = F.relu(self.output_projection1(h))                # [B, hidden_dim]
        h = F.relu(self.output_projection2(h))                # [B, hidden_dim]
        y0 = self.output_y0(F.relu(self.y0_layer(h)))        # [B, 1]
        y1 = self.output_y1(F.relu(self.y1_layer(h)))        # [B, 1]
        return torch.cat([y0, y1], dim=1)

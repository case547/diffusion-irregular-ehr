from pydantic import BaseModel


class ModelConfig(BaseModel):
    feature_dim: int
    latent_dim: int
    hidden_dim: int
    num_layers: int


class DiffusionConfig(BaseModel):
    num_steps: int
    beta_start: float | None
    beta_end: float | None
    schedule: str
    embedding_dim: int
    block_dim: int
    hidden_dim: int
    num_blocks: int
    clip_denoised: bool = False
    use_propnet: bool = False
    consistency_weight: float = 0.0
    consistency_warmup_frac: float = 0.5
    consistency_min_tau_frac: float = 0.5
    cf_anchor_weight: float = 0.0  # soft-mask weight for the counterfactual slot, anchored to
                                    # per-arm population means (0 = off, today's hard mask)


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int
    lr: float
    seed: int
    K: int
    checkpoint_dir: str


class DataConfig(BaseModel):
    path: str
    replication: int | list[int]
    train_ratio: float
    test_ratio: float
    confounder_effect: float = 0.0


class Config(BaseModel):
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    data: DataConfig

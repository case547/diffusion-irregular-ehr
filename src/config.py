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
    use_ipw: bool = False
    cf_anchor_weight: float = 0.0
    ipw_ramp_start: int = 0
    ipw_ramp_end: int = 0
    ipw_clip_prop: float = 0.1
    ipw_z_samples: int = 5
    ipw_ema_decay: float = 0.0
    a_decoder_label_smoothing: float = 0.0
    ttur_factor: float = 1.0


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

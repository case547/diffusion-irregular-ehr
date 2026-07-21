from pydantic import BaseModel


class ModelConfig(BaseModel):
    feature_dim: int
    latent_dim: int
    hidden_dim: int
    num_layers: int


class DiffusionConfig(BaseModel):
    num_steps: int
    beta_start: float
    beta_end: float
    schedule: str
    embedding_dim: int
    block_dim: int
    hidden_dim: int
    num_blocks: int


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int
    lr: float
    seed: int
    K: int
    patience: int
    warmup_epochs: int
    checkpoint_dir: str


class DataConfig(BaseModel):
    path: str
    replication: int
    train_ratio: float
    test_ratio: float


class Config(BaseModel):
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    data: DataConfig

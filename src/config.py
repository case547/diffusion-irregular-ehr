from pydantic import BaseModel, Field, model_validator


class VAEConfig(BaseModel):
    feature_dim: int
    latent_dim: int
    hidden_dim: int
    a_decoder_hidden_dim: int
    encoder_num_layers: int
    decoder_num_layers: int
    aux_num_layers: int
    a_decoder_label_smoothing: float = Field(default=0.0, ge=0.0, lt=0.5)


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
    cf_anchor_weight: float = 0.0
    use_ipw: bool = False
    ipw_ramp_start: int = 0
    ipw_ramp_end: int = 0
    ipw_clip_prop: float = 0.1
    ipw_z_samples: int = 1

    @model_validator(mode="after")
    def _validate_ipw_fields(self) -> DiffusionConfig:
        if self.use_ipw:
            assert self.ipw_ramp_end > self.ipw_ramp_start, (
                "ipw_ramp_end must be > ipw_ramp_start when use_ipw=True"
            )
            assert self.ipw_z_samples >= 1, (
                "ipw_z_samples must be >= 1 when use_ipw=True -- 0 makes torch.stack([]) "
                "raise RuntimeError inside HybridModel._compute_pi_hat"
            )
        return self


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int
    lr: float
    seed: int
    K: int
    checkpoint_dir: str
    ipw_ema_decay: float = 0.0
    prop_ttur_factor: float = 1.0


class DataConfig(BaseModel):
    path: str
    replication: int | list[int]
    train_ratio: float
    test_ratio: float
    confounder_effect: float = 0.0


class Config(BaseModel):
    vae: VAEConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    data: DataConfig

    @model_validator(mode="after")
    def _validate_ipw_ema_decay(self) -> Config:
        if self.diffusion.use_ipw:
            assert 0.0 <= self.train.ipw_ema_decay < 1.0, (
                "ipw_ema_decay must be in [0,1) when use_ipw=True -- 1.0 makes the EMA"
                " shadow an copy of the model at the ipw_ramp_start, i.e. frozen, regardless"
                " of much further training moves the live a_decoder/encoder model parameters."
            )
        return self

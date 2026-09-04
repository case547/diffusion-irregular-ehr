from pydantic import BaseModel, model_validator


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

    @model_validator(mode="after")
    def _validate_ipw_fields(self) -> DiffusionConfig:
        # Unlike the three below, this one isn't gated on use_ipw: a_decoder_label_smoothing
        # is read/applied unconditionally in HybridModel.compute_loss (see the log_pa
        # computation), so an out-of-range value is live even with use_ipw=False.
        assert 0.0 <= self.a_decoder_label_smoothing < 0.5, (
            "a_decoder_label_smoothing must be in [0, 0.5) -- at >= 0.5, "
            "a*(1-2*eps)+eps inverts the smoothed target"
        )
        if self.use_ipw:
            assert self.ipw_ramp_end > self.ipw_ramp_start, (
                "ipw_ramp_end must be > ipw_ramp_start when use_ipw=True"
            )
            assert 0.0 < self.ipw_ema_decay < 1.0, (
                "ipw_ema_decay must be in (0,1) when use_ipw=True -- 0.0 makes the EMA "
                "shadow an exact copy of the live model every step (no decoupling at all), "
                "silently defeating the mechanism's central defense against circularity"
            )
            assert self.ipw_z_samples >= 1, (
                "ipw_z_samples must be >= 1 when use_ipw=True -- 0 makes torch.stack([]) "
                "raise RuntimeError inside HybridModel._compute_phat"
            )
        return self


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

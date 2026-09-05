import pytest
from pydantic import ValidationError

from src.config import DiffusionConfig

BASE_KWARGS = dict(
    num_steps=10,
    beta_start=0.0001,
    beta_end=0.02,
    schedule="quad",
    embedding_dim=16,
    block_dim=16,
    hidden_dim=32,
    num_blocks=2,
)


def test_ipw_fields_have_inert_defaults():
    cfg = DiffusionConfig(**BASE_KWARGS)
    assert cfg.use_ipw is False
    assert cfg.ipw_ramp_start == 0
    assert cfg.ipw_ramp_end == 0
    assert cfg.ipw_clip_prop == 0.1
    assert cfg.ipw_z_samples == 1
    assert cfg.use_dr_blend is False


def test_ipw_fields_accept_overrides():
    cfg = DiffusionConfig(
        **BASE_KWARGS,
        use_ipw=True,
        ipw_ramp_start=100,
        ipw_ramp_end=300,
        ipw_clip_prop=0.05,
        ipw_z_samples=3,
    )
    assert cfg.ipw_ramp_start == 100
    assert cfg.ipw_ramp_end == 300
    assert cfg.ipw_clip_prop == 0.05
    assert cfg.ipw_z_samples == 3


def test_use_dr_blend_requires_use_ipw():
    with pytest.raises(ValidationError):
        DiffusionConfig(**BASE_KWARGS, use_ipw=False, use_dr_blend=True)


def test_use_dr_blend_allowed_when_use_ipw_true():
    cfg = DiffusionConfig(
        **BASE_KWARGS, use_ipw=True, ipw_ramp_start=0, ipw_ramp_end=1, use_dr_blend=True
    )
    assert cfg.use_dr_blend is True

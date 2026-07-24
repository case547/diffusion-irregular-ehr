import numpy as np

from experiment import run_condition
from src.config import Config, DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from src.data import CausalDataset

SMALL_CFG = Config(
    model=ModelConfig(feature_dim=5, latent_dim=4, hidden_dim=16, num_layers=2),
    diffusion=DiffusionConfig(
        num_steps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="quad",
        embedding_dim=16,
        block_dim=16,
        hidden_dim=32,
        num_blocks=2,
    ),
    train=TrainConfig(
        epochs=2,
        batch_size=8,
        lr=1e-3,
        seed=0,
        K=3,
        use_final_model=False,
        early_stopping=True,
        patience=999,
        warmup_epochs=0,
        checkpoint_dir="/tmp",
    ),
    data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
)


def _ds(n=32, f=5):
    x = np.random.randn(n, f).astype(np.float32)
    a = np.random.randint(0, 2, n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    y_cf = np.random.randn(n).astype(np.float32)
    mu0 = np.random.randn(n).astype(np.float32)
    mu1 = np.random.randn(n).astype(np.float32)
    return CausalDataset(x, a, y, y_cf, mu0, mu1)


def test_run_condition_returns_metrics():
    ds = _ds()
    result = run_condition("test_run_id", SMALL_CFG, train_ds=ds, val_ds=ds, test_ds=ds)
    assert set(result.keys()) == {
        "coverage_95_y0",
        "coverage_95_y1",
        "width_95_y0",
        "width_95_y1",
        "coverage_99_y0",
        "coverage_99_y1",
        "width_99_y0",
        "width_99_y1",
        "rmse_y0",
        "rmse_y1",
        "pehe",
    }
    for v in result.values():
        assert isinstance(v, float)
        assert np.isfinite(v)

import numpy as np

from experiment import run_condition
from src.config import Config, DataConfig, DiffusionConfig, TrainConfig, VAEConfig
from src.data import CausalDataset

SMALL_CFG = Config(
    vae=VAEConfig(
        feature_dim=5, latent_dim=4, hidden_dim=16, num_layers=2, a_decoder_hidden_dim=5
    ),
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
    train=TrainConfig(epochs=2, batch_size=8, lr=1e-3, seed=0, K=3, checkpoint_dir="/tmp"),
    data=DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15),
)


def _ds(n=32, f=5):
    x = np.random.randn(n, f).astype(np.float32)
    a = np.random.randint(0, 2, n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    y_cf = np.random.randn(n).astype(np.float32)
    mu0 = np.random.randn(n).astype(np.float32)
    mu1 = np.random.randn(n).astype(np.float32)
    conf = np.zeros(n, dtype=np.float32)
    return CausalDataset(x, a, y, y_cf, mu0, mu1, conf, y_mean=0.0, y_std=1.0)


def test_run_condition_returns_metrics():
    ds = _ds()
    result_val, result_test = run_condition(
        "test_run", SMALL_CFG, train_ds=ds, val_ds=ds, test_ds=ds
    )

    expected_keys = {
        "wasserstein_y0",
        "wasserstein_y1",
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

    assert set(result_val.keys()) == expected_keys
    assert set(result_test.keys()) == expected_keys

    for v in result_val.values():
        assert isinstance(v, float)
        assert np.isfinite(v)
    for v in result_test.values():
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_run_condition_pretrains_aux_outcome_for_hybrid_with_anchor(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "experiment.train_aux_outcome",
        lambda aux, train_loader, val_loader, cfg, device, log_fn=None: calls.append(True),
    )
    cfg = SMALL_CFG.model_copy(
        update={"diffusion": SMALL_CFG.diffusion.model_copy(update={"cf_anchor_weight": 0.1})}
    )
    ds = _ds()
    run_condition("test_pretrain_on", cfg, train_ds=ds, val_ds=ds, test_ds=ds)
    assert len(calls) == 1


def test_run_condition_skips_pretrain_when_anchor_off(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "experiment.train_aux_outcome",
        lambda aux, train_loader, val_loader, cfg, device, log_fn=None: calls.append(True),
    )
    ds = _ds()
    run_condition("test_pretrain_off", SMALL_CFG, train_ds=ds, val_ds=ds, test_ds=ds)
    assert len(calls) == 0


def test_run_condition_freezes_aux_outcome_after_pretrain(monkeypatch):
    monkeypatch.setattr(
        "experiment.train_aux_outcome",
        lambda aux, train_loader, val_loader, cfg, device, log_fn=None: None,
    )
    cfg = SMALL_CFG.model_copy(
        update={"diffusion": SMALL_CFG.diffusion.model_copy(update={"cf_anchor_weight": 0.1})}
    )
    ds = _ds()
    from src.model import HybridModel

    captured = {}
    orig_init = HybridModel.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured["model"] = self

    monkeypatch.setattr(HybridModel, "__init__", spy_init)
    run_condition("test_freeze_on", cfg, train_ds=ds, val_ds=ds, test_ds=ds)
    assert all(not p.requires_grad for p in captured["model"].aux_outcome.parameters())


def test_run_condition_leaves_aux_outcome_trainable_when_anchor_off(monkeypatch):
    monkeypatch.setattr(
        "experiment.train_aux_outcome",
        lambda aux, train_loader, val_loader, cfg, device, log_fn=None: None,
    )
    ds = _ds()
    from src.model import HybridModel

    captured = {}
    orig_init = HybridModel.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured["model"] = self

    monkeypatch.setattr(HybridModel, "__init__", spy_init)
    run_condition("test_freeze_off", SMALL_CFG, train_ds=ds, val_ds=ds, test_ds=ds)
    assert all(p.requires_grad for p in captured["model"].aux_outcome.parameters())


def test_run_condition_skips_pretrain_for_diffpo(monkeypatch):
    from src.model import DiffPO

    calls = []
    monkeypatch.setattr(
        "experiment.train_aux_outcome",
        lambda aux, train_loader, val_loader, cfg, device, log_fn=None: calls.append(True),
    )
    cfg = SMALL_CFG.model_copy(
        update={"diffusion": SMALL_CFG.diffusion.model_copy(update={"cf_anchor_weight": 0.1})}
    )
    ds = _ds()
    run_condition(
        "test_pretrain_diffpo", cfg, train_ds=ds, val_ds=ds, test_ds=ds, model_cls=DiffPO
    )
    assert len(calls) == 0

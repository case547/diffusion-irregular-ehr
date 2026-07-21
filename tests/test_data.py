import numpy as np
import pandas as pd
import pytest
import torch

from src.data import CausalDataset, load_ihdp, make_ihdp_confounded, load_acic, make_acic_confounded


# ── shared fixture ────────────────────────────────────────────────────────────

def _fake(n: int = 100, f: int = 25):
    return (
        np.random.randn(n, f).astype(np.float32),        # x
        np.random.randint(0, 2, n).astype(np.float32),   # a
        np.random.randn(n).astype(np.float32),            # y
        np.random.randn(n).astype(np.float32),            # y_cf
        np.random.randn(n).astype(np.float32),            # mu0
        np.random.randn(n).astype(np.float32),            # mu1
        np.random.randint(0, 2, n).astype(np.float32),   # confounder
    )


# ── CausalDataset ─────────────────────────────────────────────────────────────

def test_dataset_shapes():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    assert len(ds) == 100
    item = ds[0]
    assert item["x"].shape == (25,)
    assert item["a"].shape == ()
    assert item["y"].shape == ()
    assert item["mu0"].shape == ()
    assert item["mu1"].shape == ()
    assert item["x"].dtype == torch.float32


def test_dataset_no_mu():
    x, a, y, _, _, _, _ = _fake(50)
    ds = CausalDataset(x, a, y)
    item = ds[0]
    assert "mu0" not in item
    assert "mu1" not in item


def test_confounder_not_in_getitem():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    assert "confounder" not in ds[0]
    assert ds.confounder is not None


# ── IHDP confounding ──────────────────────────────────────────────────────────

def test_make_ihdp_confounded_shapes():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    ds_c = make_ihdp_confounded(ds)
    assert ds_c[0]["x"].shape == (25,)   # x unchanged


def test_make_ihdp_confounded_flip():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    ds_c = make_ihdp_confounded(ds)
    a_orig = ds.a.numpy()
    a_conf = ds_c.a.numpy()
    mask = conf == 1
    assert np.all(a_conf[mask] == 1 - a_orig[mask])
    assert np.all(a_conf[~mask] == a_orig[~mask])


def test_make_ihdp_confounded_outcomes_unchanged():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    ds_c = make_ihdp_confounded(ds)
    np.testing.assert_array_equal(ds.y.numpy(), ds_c.y.numpy())
    np.testing.assert_array_equal(ds.mu0.numpy(), ds_c.mu0.numpy())
    np.testing.assert_array_equal(ds.mu1.numpy(), ds_c.mu1.numpy())


# ── ACIC loading ──────────────────────────────────────────────────────────────

@pytest.fixture
def acic_csv(tmp_path):
    """Fake ACIC norm_data CSV: 100 rows, 5 covariates (cols 5-9)."""
    np.random.seed(0)
    N, F = 100, 5
    data = np.zeros((N, F + 5), dtype=np.float32)
    data[:50, 0] = 0; data[50:, 0] = 1          # treatment
    data[:, 1] = np.random.randn(N)              # y0
    data[:, 2] = np.random.randn(N)              # y1
    data[:, 3] = data[:, 1]                      # mu0
    data[:, 4] = data[:, 2]                      # mu1
    data[:, 5:] = np.random.randn(N, F)          # covariates
    # make col 5 strongly correlated with treatment
    data[:, 5] = data[:, 0] + 0.1 * np.random.randn(N)
    sheet_id = "fake_sheet"
    pd.DataFrame(data).to_csv(tmp_path / f"{sheet_id}.csv", index=False)
    return str(tmp_path), sheet_id, N, F


def test_load_acic_shapes(acic_csv):
    data_dir, sheet_id, N, F = acic_csv
    train_ds, val_ds, test_ds = load_acic(data_dir, sheet_id)
    assert train_ds[0]["x"].shape == (F,)
    assert train_ds[0]["a"].shape == ()
    assert train_ds[0]["y"].shape == ()
    assert train_ds[0]["mu0"].shape == ()
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == N


def test_load_acic_confounder(acic_csv):
    data_dir, sheet_id, N, F = acic_csv
    train_ds, val_ds, test_ds = load_acic(data_dir, sheet_id)
    assert train_ds.confounder is not None
    assert set(np.unique(train_ds.confounder)).issubset({0.0, 1.0})


def test_make_acic_confounded_flip(acic_csv):
    data_dir, sheet_id, N, F = acic_csv
    train_ds, _, _ = load_acic(data_dir, sheet_id)
    conf_ds = make_acic_confounded(train_ds)
    a_orig = train_ds.a.numpy()
    a_conf = conf_ds.a.numpy()
    mask = train_ds.confounder == 1
    assert np.all(a_conf[mask] == 1 - a_orig[mask])
    assert np.all(a_conf[~mask] == a_orig[~mask])


def test_make_acic_confounded_x_unchanged(acic_csv):
    data_dir, sheet_id, N, F = acic_csv
    train_ds, _, _ = load_acic(data_dir, sheet_id)
    conf_ds = make_acic_confounded(train_ds)
    np.testing.assert_array_equal(train_ds.x.numpy(), conf_ds.x.numpy())
    assert conf_ds[0]["x"].shape == (F,)

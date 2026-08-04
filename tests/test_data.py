import numpy as np
import torch

from src.data import CausalDataset, make_ihdp_confounded

# ── shared fixture ────────────────────────────────────────────────────────────


def _fake(n: int = 100, f: int = 25):
    return (
        np.random.randn(n, f).astype(np.float32),  # x
        np.random.randint(0, 2, n).astype(np.float32),  # a
        np.random.randn(n).astype(np.float32),  # y
        np.random.randn(n).astype(np.float32),  # y_cf
        np.random.randn(n).astype(np.float32),  # mu0
        np.random.randn(n).astype(np.float32),  # mu1
        np.random.randint(0, 2, n).astype(np.float32),  # confounder
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
    assert ds_c[0]["x"].shape == (25,)  # x unchanged


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

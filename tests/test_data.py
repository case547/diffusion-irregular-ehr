import numpy as np
import pytest
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
    ds_conf = make_ihdp_confounded(ds)
    assert ds_conf[0]["x"].shape == (25,)  # x unchanged


def test_make_ihdp_confounded_flip():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    ds_conf = make_ihdp_confounded(ds)
    a_orig = ds.a.numpy()
    a_conf = ds_conf.a.numpy()
    mask = conf == 1
    assert np.all(a_conf[mask] == 1 - a_orig[mask])
    assert np.all(a_conf[~mask] == a_orig[~mask])


def test_make_ihdp_confounded_outcomes_swapped_for_flipped():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)
    ds_conf = make_ihdp_confounded(ds)
    mask = conf == 1

    y_orig, y_cf_orig = ds.y.numpy(), ds.y_cf.numpy()
    y_conf, y_cf_conf = ds_conf.y.numpy(), ds_conf.y_cf.numpy()

    # Flipped subjects: y/y_cf swapped, since flipping `a` changes which PO is factual.
    np.testing.assert_array_equal(y_conf[mask], y_cf_orig[mask])
    np.testing.assert_array_equal(y_cf_conf[mask], y_orig[mask])

    # Unflipped subjects: y/y_cf unchanged.
    np.testing.assert_array_equal(y_conf[~mask], y_orig[~mask])
    np.testing.assert_array_equal(y_cf_conf[~mask], y_cf_orig[~mask])

    # mu0/mu1 identify potential outcomes by treatment arm, not factual status --
    # unaffected by which treatment ended up assigned.
    np.testing.assert_array_equal(ds.mu0.numpy(), ds_conf.mu0.numpy())
    np.testing.assert_array_equal(ds.mu1.numpy(), ds_conf.mu1.numpy())


def test_make_ihdp_confounded_outcome_effect():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    y_mean, y_std = 5.0, 2.0  # non-trivial, to exercise de/re-normalisation
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf, y_mean=y_mean, y_std=y_std)
    effect = 0.4
    ds_conf = make_ihdp_confounded(ds, effect=effect)
    mask = conf == 1

    mu0_orig_raw = ds.mu0.numpy() * y_std + y_mean
    mu1_orig_raw = ds.mu1.numpy() * y_std + y_mean
    mu0_new_raw = ds_conf.mu0.numpy() * y_std + y_mean
    mu1_new_raw = ds_conf.mu1.numpy() * y_std + y_mean

    # Direct outcome effect, checked in raw units (where Hill's mechanism is
    # defined): multiplicative for mu0, additive for mu1, only for confounder==1
    # subjects (mu0/mu1 are never touched by the flip step).
    np.testing.assert_allclose(
        mu0_new_raw[mask], mu0_orig_raw[mask] * np.exp(effect), rtol=1e-5
    )
    np.testing.assert_allclose(mu1_new_raw[mask], mu1_orig_raw[mask] + effect, rtol=1e-5)
    np.testing.assert_allclose(mu0_new_raw[~mask], mu0_orig_raw[~mask], rtol=1e-5)
    np.testing.assert_allclose(mu1_new_raw[~mask], mu1_orig_raw[~mask], rtol=1e-5)

    # Noise is preserved (in raw units), not resampled: reconstruct each subject's
    # original and new noise (accounting for the swap on flipped subjects) and
    # check they match.
    a_orig = ds.a.numpy()
    y0_old_raw = np.where(a_orig == 0, ds.y.numpy(), ds.y_cf.numpy()) * y_std + y_mean
    y1_old_raw = np.where(a_orig == 0, ds.y_cf.numpy(), ds.y.numpy()) * y_std + y_mean
    noise0_old = y0_old_raw - mu0_orig_raw
    noise1_old = y1_old_raw - mu1_orig_raw

    a_new = ds_conf.a.numpy()
    y0_new_raw = np.where(a_new == 0, ds_conf.y.numpy(), ds_conf.y_cf.numpy()) * y_std + y_mean
    y1_new_raw = np.where(a_new == 0, ds_conf.y_cf.numpy(), ds_conf.y.numpy()) * y_std + y_mean
    noise0_new = y0_new_raw - mu0_new_raw
    noise1_new = y1_new_raw - mu1_new_raw

    np.testing.assert_allclose(noise0_new, noise0_old, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(noise1_new, noise1_old, rtol=1e-4, atol=1e-4)


def test_make_ihdp_confounded_outcome_effect_requires_normalisation_stats():
    x, a, y, y_cf, mu0, mu1, conf = _fake(100)
    ds = CausalDataset(x, a, y, y_cf, mu0, mu1, conf)  # no y_mean/y_std
    with pytest.raises(AssertionError):
        make_ihdp_confounded(ds, effect=0.4)

import torch

from src.zspace_ipw import (
    calibration_diagnostic,
    effective_sample_size,
    ramp_weight,
    zspace_ipw_weight,
)


def test_zspace_ipw_weight_matches_formula_when_no_trim():
    p_hat = torch.tensor([0.5, 0.3, 0.7, 0.4])
    a = torch.tensor([1.0, 0.0, 1.0, 0.0])
    w = zspace_ipw_weight(p_hat, a, clip_prop=0.1)
    raw = a / p_hat + (1 - a) / (1 - p_hat)
    expected = raw / raw.mean()
    assert torch.allclose(w, expected)


def test_zspace_ipw_weight_trims_low_prop_treated_to_one_before_normalising():
    # subject 0: treated, p_hat=0.02 < clip_prop=0.1 -> trimmed to raw weight 1.0
    # subject 1: untreated, p_hat=0.5 -> not trimmed, raw weight 1/(1-0.5)=2.0
    p_hat = torch.tensor([0.02, 0.5])
    a = torch.tensor([1.0, 0.0])
    w = zspace_ipw_weight(p_hat, a, clip_prop=0.1)
    raw_untrimmed = torch.tensor([1.0, 2.0])
    expected = raw_untrimmed / raw_untrimmed.mean()
    assert torch.allclose(w, expected)


def test_zspace_ipw_weight_trims_high_prop_untreated_to_one():
    # subject: untreated, p_hat=0.95 -> 1-p_hat=0.05 < clip_prop=0.1 -> trimmed to 1.0
    p_hat = torch.tensor([0.95, 0.5])
    a = torch.tensor([0.0, 0.0])
    w = zspace_ipw_weight(p_hat, a, clip_prop=0.1)
    raw_untrimmed = torch.tensor([1.0, 2.0])
    expected = raw_untrimmed / raw_untrimmed.mean()
    assert torch.allclose(w, expected)


def test_zspace_ipw_weight_does_not_trim_the_safe_side():
    # treated with HIGH p_hat is never dangerous (weight -> 1 as p_hat -> 1), so a
    # treated subject with p_hat=0.95 must NOT be trimmed even though it's outside
    # a naive symmetric [clip_prop, 1-clip_prop] band.
    p_hat = torch.tensor([0.95, 0.5])
    a = torch.tensor([1.0, 0.0])
    w = zspace_ipw_weight(p_hat, a, clip_prop=0.1)
    raw = a / p_hat + (1 - a) / (1 - p_hat)  # [1/0.95, 2.0], neither trimmed
    expected = raw / raw.mean()
    assert torch.allclose(w, expected)


def test_zspace_ipw_weight_mean_is_one():
    torch.manual_seed(0)
    p_hat = torch.rand(50) * 0.8 + 0.1
    a = torch.randint(0, 2, (50,)).float()
    w = zspace_ipw_weight(p_hat, a, clip_prop=0.1)
    assert w.mean().item() == pytest_approx(1.0)


def pytest_approx(x, abs_=1e-5):
    import pytest

    return pytest.approx(x, abs=abs_)


def test_ramp_weight_is_identity_before_ramp_start():
    w = torch.tensor([2.0, 0.5])
    w_eff = ramp_weight(w, curr_epoch=5, ramp_start=100, ramp_end=200)
    assert torch.allclose(w_eff, torch.ones_like(w))


def test_ramp_weight_is_full_strength_after_ramp_end():
    w = torch.tensor([2.0, 0.5])
    w_eff = ramp_weight(w, curr_epoch=300, ramp_start=100, ramp_end=200)
    assert torch.allclose(w_eff, w)


def test_ramp_weight_interpolates_halfway():
    w = torch.tensor([3.0])
    w_eff = ramp_weight(w, curr_epoch=150, ramp_start=100, ramp_end=200)  # ramp=0.5
    assert torch.allclose(w_eff, torch.tensor([2.0]))  # 1.0 + 0.5*(3.0-1.0)


def test_effective_sample_size_full_when_weights_equal():
    w = torch.ones(10)
    assert effective_sample_size(w) == pytest_approx(10.0)


def test_effective_sample_size_drops_with_one_dominant_weight():
    w = torch.cat([torch.full((9,), 1.0), torch.tensor([50.0])])
    ess = effective_sample_size(w)
    assert ess < 5.0  # one large weight collapses ESS well below n=10


def test_calibration_diagnostic_perfect_calibration():
    # 10 subjects, p_hat exactly matches treatment status within each singleton bin
    p_hat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    a = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = calibration_diagnostic(p_hat, a, n_bins=10)
    assert out["calib_mae"] == pytest_approx(0.0, abs_=1e-6)


def test_calibration_diagnostic_returns_all_bins():
    torch.manual_seed(0)
    p_hat = torch.rand(40)
    a = torch.randint(0, 2, (40,)).float()
    out = calibration_diagnostic(p_hat, a, n_bins=4)
    for i in range(4):
        assert f"calib_bin{i}_pred" in out
        assert f"calib_bin{i}_empirical" in out
    assert "calib_mae" in out


def test_calibration_diagnostic_returns_all_bins_when_not_evenly_divisible():
    """n=32, n_bins=10: 10 does not evenly divide 32. torch.chunk would silently
    return only 8 bins here (fixed chunk_size=ceil(32/10)=4, then 8 chunks of 4
    fit) -- the exact bug that slipped past an earlier review because that task's
    original tests only used evenly-divisible sizes. A direct regression test
    belongs beside the primitive, not just transitively via a training-loop test."""
    torch.manual_seed(1)
    p_hat = torch.rand(32)
    a = torch.randint(0, 2, (32,)).float()
    out = calibration_diagnostic(p_hat, a, n_bins=10)
    for i in range(10):
        assert f"calib_bin{i}_pred" in out
        assert f"calib_bin{i}_empirical" in out
    assert "calib_mae" in out

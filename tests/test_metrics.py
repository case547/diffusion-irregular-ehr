import torch

from src.metrics import coverage, pehe, rmse, wasserstein

B, K = 20, 100


def test_wasserstein_near_zero_for_accurate_predictions():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = mu0.unsqueeze(1) + torch.randn(B, K) * 0.01
    y1 = mu1.unsqueeze(1) + torch.randn(B, K) * 0.01
    wd0, wd1 = wasserstein(y0, y1, mu0, mu1)
    assert wd0 < 0.05
    assert wd1 < 0.05


def test_wasserstein_nonnegative():
    y0, y1 = torch.randn(B, K), torch.randn(B, K)
    mu0, mu1 = torch.zeros(B), torch.zeros(B)
    wd0, wd1 = wasserstein(y0, y1, mu0, mu1)
    assert wd0 >= 0.0
    assert wd1 >= 0.0


def test_coverage_high_for_accurate_predictions():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = mu0.unsqueeze(1) + torch.randn(B, K) * 0.01
    y1 = mu1.unsqueeze(1) + torch.randn(B, K) * 0.01
    cov0, cov1, w0, w1 = coverage(y0, y1, mu0, mu1, level=0.95)
    assert cov0 > 0.9
    assert cov1 > 0.9
    assert w0 > 0.0
    assert w1 > 0.0


def test_coverage_in_unit_interval():
    y0, y1 = torch.randn(B, K), torch.randn(B, K)
    mu0, mu1 = torch.zeros(B), torch.zeros(B)
    cov0, cov1, w0, w1 = coverage(y0, y1, mu0, mu1, level=0.95)
    assert 0.0 <= cov0 <= 1.0
    assert 0.0 <= cov1 <= 1.0
    assert w0 >= 0.0
    assert w1 >= 0.0


def test_coverage_width_narrow_for_concentrated_samples():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = mu0.unsqueeze(1).expand(B, K)  # all K samples identical per subject
    y1 = mu1.unsqueeze(1).expand(B, K)
    _, _, w0, w1 = coverage(y0, y1, mu0, mu1, level=0.95)
    assert w0 < 0.01
    assert w1 < 0.01


def test_rmse_near_zero_for_accurate_predictions():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = mu0.unsqueeze(1) + torch.randn(B, K) * 0.01
    y1 = mu1.unsqueeze(1) + torch.randn(B, K) * 0.01
    r0, r1 = rmse(y0, y1, mu0, mu1)
    assert r0 < 0.1
    assert r1 < 0.1


def test_rmse_large_for_wrong_predictions():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = torch.full((B, K), 10.0)
    y1 = torch.full((B, K), -10.0)
    r0, r1 = rmse(y0, y1, mu0, mu1)
    assert r0 > 5.0
    assert r1 > 5.0


def test_pehe_near_zero_for_accurate_predictions():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = mu0.unsqueeze(1) + torch.randn(B, K) * 0.01
    y1 = mu1.unsqueeze(1) + torch.randn(B, K) * 0.01
    score = pehe(y0, y1, mu0, mu1)
    assert isinstance(score, float)
    assert score < 0.1


def test_pehe_large_for_swapped_predictions():
    mu0 = torch.zeros(B)
    mu1 = torch.ones(B)
    y0 = mu1.unsqueeze(1).expand(B, K)
    y1 = mu0.unsqueeze(1).expand(B, K)
    score = pehe(y0, y1, mu0, mu1)
    assert score > 1.5

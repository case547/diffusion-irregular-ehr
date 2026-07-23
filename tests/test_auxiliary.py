import torch

from src.auxiliary import AuxOutcome, AuxTreatment

B, F = 8, 25


def test_aux_treatment_log_prob():
    aux = AuxTreatment(feature_dim=F, hidden_dim=64, num_layers=2)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    lp = aux.log_prob(x, a)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_aux_treatment_sample():
    aux = AuxTreatment(feature_dim=F, hidden_dim=64, num_layers=2)
    a_s = aux.sample(torch.randn(B, F))
    assert a_s.shape == (B,)
    assert set(a_s.int().tolist()).issubset({0, 1})


def test_aux_outcome_log_prob():
    aux = AuxOutcome(feature_dim=F, hidden_dim=64, num_layers=2)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y = torch.randn(B)
    lp = aux.log_prob(x, a, y)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_aux_outcome_sample():
    aux = AuxOutcome(feature_dim=F, hidden_dim=64, num_layers=2)
    x = torch.randn(B, F)
    a = torch.randint(0, 2, (B,)).float()
    y_s = aux.sample(x, a)
    assert y_s.shape == (B,)
    assert torch.isfinite(y_s).all()


def test_aux_outcome_tarnet_split():
    aux = AuxOutcome(feature_dim=F, hidden_dim=64, num_layers=2)
    x = torch.randn(B, F)
    y0 = aux.sample(x, torch.zeros(B))
    y1 = aux.sample(x, torch.ones(B))
    assert not torch.allclose(y0, y1)

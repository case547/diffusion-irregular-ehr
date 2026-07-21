import torch

from src.decoders import ADecoder, XDecoder

B, F, D = 8, 25, 20


def test_x_decoder_shapes():
    dec = XDecoder(latent_dim=D, feature_dim=F, hidden_dim=64, num_layers=2)
    z = torch.randn(B, D)
    loc, scale = dec(z)
    assert loc.shape == (B, F)
    assert scale.shape == (B, F)
    assert (scale > 0).all()


def test_x_decoder_log_prob():
    dec = XDecoder(latent_dim=D, feature_dim=F, hidden_dim=64, num_layers=2)
    z, x = torch.randn(B, D), torch.randn(B, F)
    lp = dec.log_prob(z, x)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_a_decoder_shapes():
    dec = ADecoder(latent_dim=D, hidden_dim=64, num_layers=2)
    logits = dec(torch.randn(B, D))
    assert logits.shape == (B,)


def test_a_decoder_log_prob():
    dec = ADecoder(latent_dim=D, hidden_dim=64, num_layers=2)
    z = torch.randn(B, D)
    a = torch.randint(0, 2, (B,)).float()
    lp = dec.log_prob(z, a)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()

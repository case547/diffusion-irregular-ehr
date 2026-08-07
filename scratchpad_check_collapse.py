"""Check DiffPOCEVAE's latent z for posterior collapse on the hybrid_conf checkpoint.

If the encoder has collapsed to the prior N(0,I), mu will sit near 0 and sigma near 1
for most latent dimensions, with little variation across subjects -- meaning z carries
almost no information about (x, a, y_fac), regardless of latent_dim/hidden_dim size.
"""

import torch

from src.config import DiffusionConfig, ModelConfig
from src.data import load_ihdp, make_ihdp_confounded
from src.model import DiffPOCEVAE

MODEL_CFG = ModelConfig(feature_dim=25, latent_dim=20, hidden_dim=64, num_layers=2)
DIFF_CFG = DiffusionConfig(
    num_steps=100,
    beta_start=0.0001,
    beta_end=0.2,
    schedule="quad",
    embedding_dim=32,
    block_dim=32,
    hidden_dim=32,
    num_blocks=4,
    clip_denoised=True,
)
CKPT_PATH = "checkpoints/final_model_hybrid_conf_2026-08-05T10_47_30.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds, y_std = load_ihdp(
    "data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15
)
train_ds, val_ds, test_ds = (make_ihdp_confounded(ds) for ds in (train_ds, val_ds, test_ds))

model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

x, a, y = train_ds.x.to(device), train_ds.a.to(device), train_ds.y.to(device)

with torch.no_grad():
    _, mu, sigma = model.encoder.rsample(x, a, y)

mu, sigma = mu.cpu(), sigma.cpu()

print(f"z shape: {mu.shape}  (N={mu.shape[0]}, latent_dim={mu.shape[1]})\n")
print("Prior is N(0, 1) per dimension. Collapse looks like mu~0, sigma~1, low mu variance.\n")

print(f"{'dim':>4} {'mean|mu|':>10} {'std(mu)':>10} {'mean sigma':>11}")
for d in range(mu.shape[1]):
    print(
        f"{d:>4}"
        f" {mu[:, d].abs().mean().item():>10.4f}"
        f" {mu[:, d].std().item():>10.4f}"
        f" {sigma[:, d].mean().item():>11.4f}"
    )

print()
print(
    f"Aggregate: mean|mu|={mu.abs().mean().item():.4f}  "
    f"mean std(mu) across dims={mu.std(dim=0).mean().item():.4f}  "
    f"mean sigma={sigma.mean().item():.4f}"
)
print(
    f"KL from prior (mean over dims and subjects): "
    f"{(0.5 * (mu.pow(2) + sigma.pow(2) - 2 * sigma.log() - 1)).mean().item():.4f}"
)

"""Probe: does z actually encode the hidden confounder (momblack)?

Trains a logistic regression on z (the trained hybrid_conf encoder's posterior mean)
to predict momblack, and compares against the same probe trained directly on x --
the raw covariates z was derived from. If x predicts momblack better than z does,
the encoder is losing confounder-relevant signal during encoding, not just failing
to have any (posterior collapse, already ruled out separately).
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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
CKPT_PATH = "checkpoints/final_model_hybrid_conf_2026-08-04T12_59_25.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds, y_std = load_ihdp(
    "data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15
)
train_ds, val_ds, test_ds = (make_ihdp_confounded(ds) for ds in (train_ds, val_ds, test_ds))

model = DiffPOCEVAE(MODEL_CFG, DIFF_CFG).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()


def get_mu(ds) -> np.ndarray:
    x, a, y = ds.x.to(device), ds.a.to(device), ds.y.to(device)
    with torch.no_grad():
        _, mu, _ = model.encoder.rsample(x, a, y)
    return mu.cpu().numpy()


x_train, x_test = train_ds.x.numpy(), test_ds.x.numpy()
z_train, z_test = get_mu(train_ds), get_mu(test_ds)
conf_train = train_ds.confounder.astype(int)
conf_test = test_ds.confounder.astype(int)

base_rate = conf_test.mean()
majority_acc = max(base_rate, 1 - base_rate)
print(f"N train={len(conf_train)}  N test={len(conf_test)}  test base rate={base_rate:.4f}")
print(f"Majority-class baseline accuracy: {majority_acc:.4f}\n")

print(f"{'probe input':<12} {'accuracy':>10} {'AUC':>10}")
for name, xtr, xte in (("x (25-dim)", x_train, x_test), ("z (20-dim)", z_train, z_test)):
    clf = LogisticRegression(max_iter=2000).fit(xtr, conf_train)
    pred = clf.predict(xte)
    proba = clf.predict_proba(xte)[:, 1]
    acc = accuracy_score(conf_test, pred)
    auc = roc_auc_score(conf_test, proba)
    print(f"{name:<12} {acc:>10.4f} {auc:>10.4f}")

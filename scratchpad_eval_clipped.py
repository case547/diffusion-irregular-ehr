"""Re-evaluate existing checkpoints with clip_denoised enabled, no retraining.

Loads each of the three (L, beta_end) checkpoints from the 2026-08-03 unclipped
sweep and re-runs evaluate() with clip_val computed the same way experiment.py
does (2 * max abs [y0,y1] over the training split), reusing the trained weights.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from src.data import load_ihdp
from src.model import DiffPO, _DiffusionBase
from train import evaluate

MODEL_CFG = ModelConfig(feature_dim=25, latent_dim=20, hidden_dim=64, num_layers=2)
TRAIN_CFG = TrainConfig(
    epochs=500,
    batch_size=256,
    lr=0.0005,
    seed=42,
    K=50,
    use_final_model=True,
    early_stopping=False,
    patience=20,
    warmup_epochs=50,
    checkpoint_dir="checkpoints",
)
DATA_CFG = DataConfig(path="data/ihdp", replication=1, train_ratio=0.7, test_ratio=0.15)

RUNS = [
    (100, 0.5, "checkpoints/final_model_naive_full_2026-08-03T16_08_35.pth"),
    (100, 0.2, "checkpoints/final_model_naive_full_2026-08-03T16_14_47.pth"),
    (150, 0.15, "checkpoints/final_model_naive_full_2026-08-03T16_19_01.pth"),
    (200, 0.1, "checkpoints/final_model_naive_full_2026-08-03T16_21_12.pth"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds, y_std = load_ihdp(
    DATA_CFG.path,
    replication=DATA_CFG.replication,
    train_ratio=DATA_CFG.train_ratio,
    test_ratio=DATA_CFG.test_ratio,
)
val_loader = DataLoader(val_ds, batch_size=TRAIN_CFG.batch_size)
test_loader = DataLoader(test_ds, batch_size=TRAIN_CFG.batch_size)

y_both = _DiffusionBase._assemble_yboth(train_ds.a, train_ds.y, train_ds.y_cf)
clip_value = 2 * y_both.abs().max().item()
print(f"y_std={y_std:.4f}  clip_value={clip_value:.4f}\n")

header = (
    f"{'config':<22} {'rmse_y0':>9} {'rmse_y1':>9} {'pehe':>9} "
    f"{'width95_y0':>11} {'width95_y1':>11} {'cov95_y0':>9} {'cov95_y1':>9}"
)

results = {}

print(f"=== VALIDATION (clip_denoised=True) ===\n{header}")
for num_steps, beta_end, ckpt_path in RUNS:
    diff_cfg = DiffusionConfig(
        num_steps=num_steps,
        beta_start=0.0001,
        beta_end=beta_end,
        schedule="quad",
        embedding_dim=32,
        block_dim=32,
        hidden_dim=32,
        num_blocks=4,
        clip_denoised=True,
    )
    model = DiffPO(MODEL_CFG, diff_cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    torch.manual_seed(0)
    result_val = evaluate(model, val_loader, TRAIN_CFG.K, device, clip_val=clip_value)
    for k in (
        "pehe",
        "rmse_y0",
        "rmse_y1",
        "width_95_y0",
        "width_95_y1",
        "width_99_y0",
        "width_99_y1",
    ):
        result_val[k] *= y_std

    name = f"L={num_steps}, beta_end={beta_end:.2f}"
    results[name] = result_val
    print(
        f"{name:<22} {result_val['rmse_y0']:>9.4f} {result_val['rmse_y1']:>9.4f} "
        f"{result_val['pehe']:>9.4f} {result_val['width_95_y0']:>11.4f} "
        f"{result_val['width_95_y1']:>11.4f} {result_val['coverage_95_y0']:>9.4f} "
        f"{result_val['coverage_95_y1']:>9.4f}"
    )

with open(Path("results") / "results_naive_full_clipping_investigation.json", "w") as f:
    json.dump(results, f, indent=2)

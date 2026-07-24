"""2×2 confounding experiment: DiffPO and DiffPO-CEVAE × full and confounded IHDP."""

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from src.config import Config
from src.data import CausalDataset, load_ihdp, make_ihdp_confounded
from src.model import DiffPO, DiffPOCEVAE, _DiffusionBase
from src.propensity import PropensityNet
from train import _train_loop, evaluate

logger = logging.getLogger(__name__)


def run_condition(
    condition: str,
    cfg: Config,
    train_ds: CausalDataset,
    val_ds: CausalDataset,
    test_ds: CausalDataset,
    model_cls: type[_DiffusionBase] = DiffPOCEVAE,
    propnet: PropensityNet | None = None,
    log_fn: Callable | None = None,
) -> dict[str, float]:
    """Train one model on one dataset condition and return test metrics."""
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)

    model = model_cls(cfg.model, cfg.diffusion).to(device)
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        cfg.train.checkpoint_dir, f"best_model_{condition}_{datetime.now().isoformat()}.pth"
    )

    _train_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        ckpt_path,
        log_fn=log_fn,
        propnet=propnet,
        early_stopping=cfg.train.early_stopping,
    )
    result = evaluate(model, test_loader, cfg.train.K, device)
    logger.info("Test results:\n%s", ", ".join(f"{k}: {v:.4f}" for k, v in result.items()))
    return result


def _fit_propnet(
    cfg: Config, *datasets: CausalDataset, log_fn: Callable | None = None
) -> PropensityNet:
    """Fit propnet on all data (train+val+test concatenated), matching DiffPO paper."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_x = torch.cat([ds.x for ds in datasets])
    all_a = torch.cat([ds.a for ds in datasets])
    propnet = PropensityNet(n_unit_in=cfg.model.feature_dim, device=device)
    propnet.fit(all_x, all_a, log_fn=log_fn)
    logger.info("PropensityNet fitted on all data: train+val+test.")
    propnet.eval()
    for p in propnet.parameters():
        p.requires_grad_(False)
    return propnet


CONDITION_MAP = {
    "diffpo_full": (DiffPO, False),
    "diffpo_conf": (DiffPO, True),
    "hybrid_full": (DiffPOCEVAE, False),
    "hybrid_conf": (DiffPOCEVAE, True),
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(CONDITION_MAP), required=True)
    parser.add_argument("--config", default="config/ihdp.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join("logs", f"{args.condition}_{datetime.now().isoformat()}.log")
            ),
        ],
    )

    with open(args.config) as f:
        cfg = Config.model_validate(yaml.safe_load(f))

    train_ds, val_ds, test_ds, y_std = load_ihdp(
        cfg.data.path,
        replication=cfg.data.replication,
        train_ratio=cfg.data.train_ratio,
        test_ratio=cfg.data.test_ratio,
    )

    model_cls, use_conf = CONDITION_MAP[args.condition]
    if use_conf:
        train_ds, val_ds, test_ds = (
            make_ihdp_confounded(ds) for ds in (train_ds, val_ds, test_ds)
        )

    with wandb.init(
        project="diffusion-irregular-ehr",
        id=f"{args.condition}_{datetime.now().isoformat()}",
        config=cfg.model_dump(),
        reinit=True,
    ) as run:
        run.define_metric("propensity/*", step_metric="propnet/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("val/*", step_metric="train/step")

        propnet = None
        if model_cls is DiffPO:
            propnet = _fit_propnet(
                cfg,
                train_ds,
                val_ds,
                test_ds,
                log_fn=lambda d, step: run.log({**d, "propnet/step": step}),
            )

        result = run_condition(
            args.condition,
            cfg,
            train_ds,
            val_ds,
            test_ds,
            model_cls,
            propnet,
            log_fn=lambda d, step: run.log({**d, "train/step": step}),
        )
        for k in (
            "pehe",
            "rmse_y0",
            "rmse_y1",
            "width_95_y0",
            "width_95_y1",
            "width_99_y0",
            "width_99_y1",
        ):
            result[k] *= y_std
        run.log({f"test/{k}": v for k, v in result.items()})

    with open(
        os.path.join("results", f"results_{args.condition}_{datetime.now().isoformat()}.json"),
        "w",
    ) as f:
        json.dump(result, f, indent=2)
    print(result)

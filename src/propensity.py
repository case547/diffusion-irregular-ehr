"""PropensityNet: P(a=1|x) classifier for DiffPO IPW weighting.

Adapted from DiffPO/PropensityNet.py: same architecture (Linear → BatchNorm → ELU,
2-class Softmax), device-parameterised, prints replaced with logger, wandb integrated.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

EPS = 1e-8

NONLIN = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
}


class PropensityNet(nn.Module):
    """P(a=1|x) classifier for IPW weighting in DiffPO.

    Same architecture as DiffPO's PropensityNet. Pre-train with .fit(), freeze,
    then pass to DiffPO.compute_loss() as propnet.
    """

    def __init__(
        self,
        n_unit_in: int,
        n_units_out_prop: int = 100,
        n_layers_out_prop: int = 0,
        nonlin: str = "elu",
        lr: float = 0.0001,
        weight_decay: float = 1e-4,
        n_iter: int = 1000,
        batch_size: int = 100,
        seed: int = 42,
        val_split_prop: float = 0.3,
        patience: int = 10,
        n_iter_min: int = 200,
        clipping_value: int = 1,
        batch_norm: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if nonlin not in NONLIN:
            raise ValueError(f"Unknown nonlinearity: {nonlin}")
        NL = NONLIN[nonlin]
        self.device = device
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.clipping_value = clipping_value

        if batch_norm:
            layers = [nn.Linear(n_unit_in, n_units_out_prop), nn.BatchNorm1d(n_units_out_prop), NL()]
        else:
            layers = [nn.Linear(n_unit_in, n_units_out_prop), NL()]

        for _ in range(n_layers_out_prop - 1):
            if dropout:
                layers.append(nn.Dropout(dropout_prob))
            if batch_norm:
                layers.extend([nn.Linear(n_units_out_prop, n_units_out_prop),
                                nn.BatchNorm1d(n_units_out_prop), NL()])
            else:
                layers.extend([nn.Linear(n_units_out_prop, n_units_out_prop), NL()])

        layers.extend([nn.Linear(n_units_out_prop, 2), nn.Softmax(dim=-1)])
        self.model = nn.Sequential(*layers).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.to(self.device))

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        return nn.NLLLoss()(torch.log(y_pred + EPS), y_target.to(self.device))

    def get_importance_weights(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """IPW weights: w/p + (1-w)/(1-p) where p = P(a=1|x). Shape (B,)."""
        p = self.forward(X)[:, 1]
        w = w.to(self.device)
        return w / (p + EPS) + (1 - w) / (1 - p + EPS)

    def fit(self, X: torch.Tensor, y: torch.Tensor, wandb_run=None) -> "PropensityNet":
        self.train()
        X = X.float().to(self.device)
        y = y.long().to(self.device)
        if self.val_split_prop > 0:
            train_idx, val_idx = train_test_split(
                np.arange(X.shape[0]),
                test_size=self.val_split_prop, random_state=self.seed, stratify=y.cpu().numpy(),
            )
            X, X_val = X[train_idx], X[val_idx]
            y, y_val = y[train_idx], y[val_idx]
        else:
            X_val, y_val = X, y

        batch_size = min(self.batch_size, X.shape[0])
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        val_loss_best = float("inf")
        patience_left = self.patience

        for i in range(self.n_iter):
            train_losses = []
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                batch_loss = self.loss(self.forward(X_batch), y_batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
                self.optimizer.step()
                train_losses.append(batch_loss.item())

            with torch.no_grad():
                val_loss = self.loss(self.forward(X_val), y_val).item()
            train_loss_avg = sum(train_losses) / len(train_losses)
            logger.info(f"PropensityNet iter {i}: train={train_loss_avg:.4f} val={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"propensity/train_loss": train_loss_avg,
                                "propensity/val_loss": val_loss}, step=i)
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                patience_left = self.patience
            else:
                patience_left -= 1
            if patience_left == 0 and i >= self.n_iter_min:
                logger.info(f"PropensityNet early stopping at iter {i}")
                break

        return self

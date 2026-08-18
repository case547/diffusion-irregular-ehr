from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    """Dataset for IHDP.

    confounder: optional binary numpy array -- not a model input, not in __getitem__.
    y_cf: noisy counterfactual outcome -- passed to denoiser input (not used in loss).
    y_mean/y_std: training-split outcome normalisation stats (set by load_ihdp), used
        by make_ihdp_confounded to apply Hill's response-surface mechanism in raw-
        outcome units. Not model inputs, not in __getitem__.
    """

    def __init__(
        self,
        x: np.ndarray,
        a: np.ndarray,
        y: np.ndarray,
        y_cf: np.ndarray | None = None,
        mu0: np.ndarray | None = None,
        mu1: np.ndarray | None = None,
        confounder: np.ndarray | None = None,
        y_mean: float | None = None,
        y_std: float | None = None,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.y_cf = torch.tensor(y_cf, dtype=torch.float32) if y_cf is not None else None
        self.mu0 = torch.tensor(mu0, dtype=torch.float32) if mu0 is not None else None
        self.mu1 = torch.tensor(mu1, dtype=torch.float32) if mu1 is not None else None
        self.confounder = confounder
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"x": self.x[idx], "a": self.a[idx], "y": self.y[idx]}
        if self.y_cf is not None:
            item["y_cf"] = self.y_cf[idx]
        if self.mu0 is not None and self.mu1 is not None:
            item["mu0"] = self.mu0[idx]
            item["mu1"] = self.mu1[idx]
        return item


def load_ihdp(
    data_dir: str,
    replication: int = 1,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
):
    """
    Load one replication from data_dir/full/ihdp_full_{replication}.csv.

    This is the full 985-subject IHDP population (~38% treated), reconstructed by
    data/ihdp/make_full_ihdp.py to keep Hill's response-surface-B data-generating
    process while including all treated infants -- unlike Hill's NPCI benchmark,
    which excludes every treated infant with a non-white mother (747 subjects,
    ~19% treated), inducing a treatment imbalance responsible for catastrophic
    diffusion-model failure on the standard benchmark.

    CSV columns (header row present):
        treat, y_factual, y_cfactual, mu0, mu1,
        bw..was (x1-x25), momwhite, momblack, momhisp.
    x[:,13] (the `first` variable) is stored as {1,2} -- adjusted to {1,0}.
    momblack is stored as ds.confounder (binary; never in x).
    Split:
        train (train_ratio) vs valtest (1-train_ratio), then test (test_ratio) from valtest;
        val = 1 - train_ratio - test_ratio -> 70/15/15 (random_state=1).
    Both splits stratified on treatment to preserve ~38% treated rate in each fold.
    Outcomes normalised to training-split mean/std.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    path = Path(data_dir) / "full" / f"ihdp_full_{replication}.csv"
    df = pd.read_csv(path)

    a = df["treat"].values.astype(np.float32)
    y = df["y_factual"].values.astype(np.float32)
    y_cf = df["y_cfactual"].values.astype(np.float32)
    mu0 = df["mu0"].values.astype(np.float32)
    mu1 = df["mu1"].values.astype(np.float32)
    x = df.iloc[:, 5:30].values.astype(np.float32)  # x1-x25
    x[:, 13] = 2 - x[:, 13]  # first: {1,2} -> {1,0}
    confounder = df["momblack"].values.astype(np.float32)

    idx = np.arange(len(a))
    valtest_ratio = 1.0 - train_ratio
    idx_train, idx_valtest = train_test_split(
        idx, test_size=valtest_ratio, random_state=1, stratify=a
    )
    idx_val, idx_test = train_test_split(
        idx_valtest,
        test_size=test_ratio / valtest_ratio,
        random_state=1,
        stratify=a[idx_valtest],
    )

    y_mean = y[idx_train].mean()
    ytrain_std = y[idx_train].std() + 1e-8
    y = (y - y_mean) / ytrain_std
    y_cf = (y_cf - y_mean) / ytrain_std
    mu0 = (mu0 - y_mean) / ytrain_std
    mu1 = (mu1 - y_mean) / ytrain_std

    def _make(idx_):
        return CausalDataset(
            x[idx_],
            a[idx_],
            y[idx_],
            y_cf[idx_],
            mu0[idx_],
            mu1[idx_],
            confounder[idx_],
            float(y_mean),
            float(ytrain_std),
        )

    return _make(idx_train), _make(idx_val), _make(idx_test), float(ytrain_std)


def make_ihdp_confounded(ds: CausalDataset, effect: float = 0.0) -> CausalDataset:
    """Confound on ds.confounder == 1 (momblack). x unchanged.

    Two independent mechanisms, applied in sequence:

    1. Direct outcome effect (skipped entirely if effect == 0): momblack shifts the
       true potential outcomes by the same rule Hill's response surface B uses for
       every other covariate -- multiplicative for mu0 (which lives in log-space,
       mu0 = exp(beta.X)), additive for mu1 (mu1 = beta.X). This asymmetry matters:
       a *shared* level shift (mu0 += c, mu1 += c) would cancel exactly out of
       mu1-mu0, leaving PEHE unbiased (only rmse_y0/rmse_y1 would move) -- using
       Hill's own asymmetric structure instead avoids that trap for free.

       ds.y_mean/ds.y_std (set by load_ihdp) are required whenever effect != 0.
       load_ihdp normalises mu0/mu1/y/y_cf to the training split's scale, but Hill's
       mechanism is defined in raw-outcome units (the {0,0.1,0.2,0.3,0.4} coefficient
       grid). Applying it directly to normalised mu0 is wrong: normalised mu0 is
       negative for ~90% of subjects (control outcomes sit below the pooled mean), so
       multiplying by exp(effect) > 1 makes it *more* negative -- decreasing raw mu0
       for most subjects, the opposite of Hill's mechanism. Fixed by de-normalising
       to raw units, applying the shift there, then re-normalising back.

       Each subject's original noise realisation (y0-mu0, y1-mu1), computed in raw
       units, is preserved and re-applied to the shifted means, rather than
       resampling fresh noise, so the only thing that changes is the systematic
       shift, not also unrelated randomness.

    2. Selection effect: treatment is flipped for confounder==1 subjects (momblack
       is not in x1-x25, but is partially recoverable via proxy variables).

       Flipping treatment changes which potential outcome is factual vs
       counterfactual, so y/y_cf are swapped for flipped subjects to stay
       consistent with the new a. mu0/mu1 (identified by treatment arm, not
       factual status) are unaffected by this step.
    """
    assert ds.confounder is not None, "ds.confounder is None; load via load_ihdp"
    conf = ds.confounder
    a_orig = ds.a.numpy()
    y = ds.y.numpy()
    y_cf = ds.y_cf.numpy() if ds.y_cf is not None else None
    mu0 = ds.mu0.numpy() if ds.mu0 is not None else None
    mu1 = ds.mu1.numpy() if ds.mu1 is not None else None

    if effect != 0.0:
        assert mu0 is not None and mu1 is not None and y_cf is not None, (
            "effect != 0 requires mu0, mu1, and y_cf; load via load_ihdp"
        )
        assert ds.y_mean is not None and ds.y_std is not None, (
            "effect != 0 requires ds.y_mean/ds.y_std (set by load_ihdp) to apply "
            "Hill's mechanism in raw-outcome units, not the normalised training scale"
        )
        y_mean, y_std = ds.y_mean, ds.y_std

        def denorm(v: np.ndarray) -> np.ndarray:
            return v * y_std + y_mean

        def renorm(v: np.ndarray) -> np.ndarray:
            return (v - y_mean) / y_std

        y0_raw = denorm(np.where(a_orig == 0, y, y_cf))
        y1_raw = denorm(np.where(a_orig == 0, y_cf, y))
        mu0_raw = denorm(mu0)
        mu1_raw = denorm(mu1)
        noise0 = y0_raw - mu0_raw
        noise1 = y1_raw - mu1_raw

        mu0_raw_new = mu0_raw * np.exp(effect * conf)
        mu1_raw_new = mu1_raw + effect * conf
        y0_raw_new = mu0_raw_new + noise0
        y1_raw_new = mu1_raw_new + noise1

        mu0 = renorm(mu0_raw_new)
        mu1 = renorm(mu1_raw_new)
        y0_new = renorm(y0_raw_new)
        y1_new = renorm(y1_raw_new)
        y = np.where(a_orig == 0, y0_new, y1_new)
        y_cf = np.where(a_orig == 0, y1_new, y0_new)

    flip = conf == 1
    a = a_orig.copy()
    a[flip] = 1.0 - a[flip]
    if y_cf is not None:
        y, y_cf = np.where(flip, y_cf, y), np.where(flip, y, y_cf)

    return CausalDataset(ds.x.numpy(), a, y, y_cf, mu0, mu1, conf.copy(), ds.y_mean, ds.y_std)

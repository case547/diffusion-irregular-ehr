import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    """Shared dataset for IHDP and ACIC.

    confounder: optional binary numpy array -- not a model input, not in __getitem__.
    y_cf: noisy counterfactual outcome -- passed to denoiser input (not used in loss).
    """

    def __init__(self, x, a, y, y_cf=None, mu0=None, mu1=None, confounder=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.y_cf = torch.tensor(y_cf, dtype=torch.float32) if y_cf is not None else None
        self.mu0 = torch.tensor(mu0, dtype=torch.float32) if mu0 is not None else None
        self.mu1 = torch.tensor(mu1, dtype=torch.float32) if mu1 is not None else None
        self.confounder = confounder

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item = {"x": self.x[idx], "a": self.a[idx], "y": self.y[idx]}
        if self.y_cf is not None:
            item["y_cf"] = self.y_cf[idx]
        if self.mu0 is not None:
            item["mu0"] = self.mu0[idx]
            item["mu1"] = self.mu1[idx]
        return item


def load_ihdp(data_dir: str, replication: int = 1, train_ratio: float = 0.7, test_ratio: float = 0.15):
    """
    Load one NPCI replication from data_dir/with_race/ihdp_with_race_{replication}.csv.

    CSV columns (header row present): treat, y_factual, y_cfactual, mu0, mu1,
    bw..was (x1-x25), momwhite, momblack, momhisp.
    x[:,13] (the `first` variable) is stored as {1,2} -- adjusted to {1,0}.
    momblack is stored as ds.confounder (binary; never in x).
    Split: train (train_ratio) vs valtest (1-train_ratio), then test (test_ratio) from valtest;
    val = 1 - train_ratio - test_ratio -> 70/15/15 (random_state=1).
    Both splits stratified on treatment to preserve ~20% treated rate in each fold.
    Outcomes normalised to training-split mean/std.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    path = os.path.join(data_dir, "with_race", f"ihdp_with_race_{replication}.csv")
    df = pd.read_csv(path)

    a = df["treat"].values.astype(np.float32)
    y = df["y_factual"].values.astype(np.float32)
    y_cf = df["y_cfactual"].values.astype(np.float32)
    mu0 = df["mu0"].values.astype(np.float32)
    mu1 = df["mu1"].values.astype(np.float32)
    x = df.iloc[:, 5:30].values.astype(np.float32)   # x1-x25
    x[:, 13] = 2 - x[:, 13]                            # first: {1,2} -> {1,0}
    confounder = df["momblack"].values.astype(np.float32)

    idx = np.arange(len(a))
    valtest_ratio = 1.0 - train_ratio
    idx_train, idx_valtest = train_test_split(idx, test_size=valtest_ratio, random_state=1, stratify=a)
    idx_val, idx_test = train_test_split(idx_valtest, test_size=test_ratio / valtest_ratio, random_state=1, stratify=a[idx_valtest])

    y_mean = y[idx_train].mean()
    y_std = y[idx_train].std() + 1e-8
    y = (y - y_mean) / y_std
    y_cf = (y_cf - y_mean) / y_std
    mu0 = (mu0 - y_mean) / y_std
    mu1 = (mu1 - y_mean) / y_std

    def _make(idx_):
        return CausalDataset(x[idx_], a[idx_], y[idx_], y_cf[idx_], mu0[idx_], mu1[idx_], confounder[idx_])

    return _make(idx_train), _make(idx_val), _make(idx_test)


def make_ihdp_confounded(ds: CausalDataset) -> CausalDataset:
    """Flip treatment where ds.confounder == 1 (momblack). x unchanged.

    momblack is not in x1-x25, but is partially recoverable via proxy variables
    (site indicators, maternal education). Outcomes and ground-truth POs unchanged.
    """
    assert ds.confounder is not None, "ds.confounder is None; load via load_ihdp"
    a = ds.a.numpy().copy()
    a[ds.confounder == 1] = 1.0 - a[ds.confounder == 1]
    y_cf = ds.y_cf.numpy() if ds.y_cf is not None else None
    mu0 = ds.mu0.numpy() if ds.mu0 is not None else None
    mu1 = ds.mu1.numpy() if ds.mu1 is not None else None
    return CausalDataset(ds.x.numpy(), a, ds.y.numpy(), y_cf, mu0, mu1, ds.confounder)


def load_acic(data_dir: str, sheet_id: str, split_seed: int = 1, train_ratio: float = 0.7, test_ratio: float = 0.15):
    """
    Load one ACIC2018 sheet from data_dir/{sheet_id}.csv (DiffPO norm_data format).

    CSV has no named columns: [0: a, 1: y0, 2: y1, 3: mu0, 4: mu1, 5+: x].
    Factual outcome: y = (1-a)*y0 + a*y1.
    Confounder: covariate most correlated with treatment, binarized at its median.
    The confounder IS in x -- DiffPO can observe it directly; DiffPO-CEVAE
    additionally captures it via the latent z path.
    Split: train (train_ratio) vs valtest (1-train_ratio), then test (test_ratio) from valtest;
    val = 1 - train_ratio - test_ratio -> 70/15/15 (random_state=split_seed).
    Both splits stratified on treatment to preserve treatment rate in each fold.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    path = os.path.join(data_dir, f"{sheet_id}.csv")
    data = pd.read_csv(path).values.astype(np.float32)

    a = data[:, 0]
    y0_pot = data[:, 1]
    y1_pot = data[:, 2]
    mu0 = data[:, 3]
    mu1 = data[:, 4]
    x = data[:, 5:]
    y = (1.0 - a) * y0_pot + a * y1_pot          # factual outcome
    y_cf = a * y0_pot + (1.0 - a) * y1_pot       # counterfactual outcome

    # covariate most correlated with treatment -> binary confounder
    corr = np.abs(np.corrcoef(x.T, a)[-1, :-1])   # shape (F,)
    hidden_idx = int(corr.argmax())
    confounder = (x[:, hidden_idx] > np.median(x[:, hidden_idx])).astype(np.float32)

    idx = np.arange(len(a))
    valtest_ratio = 1.0 - train_ratio
    idx_train, idx_valtest = train_test_split(idx, test_size=valtest_ratio, random_state=split_seed, stratify=a)
    idx_val, idx_test = train_test_split(idx_valtest, test_size=test_ratio / valtest_ratio, random_state=split_seed, stratify=a[idx_valtest])

    def _make(idx_):
        return CausalDataset(x[idx_], a[idx_], y[idx_], y_cf[idx_], mu0[idx_], mu1[idx_], confounder[idx_])

    return _make(idx_train), _make(idx_val), _make(idx_test)


def make_acic_confounded(ds: CausalDataset) -> CausalDataset:
    """Flip treatment where ds.confounder == 1. x unchanged.

    The confounder was selected as the covariate most correlated with treatment in the
    original data. Both DiffPO and DiffPO-CEVAE can observe it in x, but DiffPO-CEVAE
    may learn a cleaner latent representation of its causal role.
    """
    assert ds.confounder is not None, "ds.confounder is None; load via load_acic"
    a = ds.a.numpy().copy()
    a[ds.confounder == 1] = 1.0 - a[ds.confounder == 1]
    y_cf = ds.y_cf.numpy() if ds.y_cf is not None else None
    mu0 = ds.mu0.numpy() if ds.mu0 is not None else None
    mu1 = ds.mu1.numpy() if ds.mu1 is not None else None
    return CausalDataset(ds.x.numpy(), a, ds.y.numpy(), y_cf, mu0, mu1, ds.confounder)

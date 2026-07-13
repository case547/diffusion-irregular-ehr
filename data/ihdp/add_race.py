from typing import cast

import pandas as pd
import pyreadr

# fmt: off
NPCI_COLS = [
    "treat",
    "y_factual","y_cfactual", "mu0", "mu1",
    # Continuous variables
    "bw", "b.head", "preterm", "birth.o", "nnhealth", "momage",
    # Binary variables
    "sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll",
    "cig", "first", "booze", "drugs", "work.dur", "prenatal",
    # Site indicators
    "ark", "ein", "har", "mia", "pen", "tex", "was",
]
# fmt: on

result = pyreadr.read_r("data/ihdp/sim.data")
imp1: pd.DataFrame = result["imp1"]
print(f"imp1 shape: {imp1.shape}, columns: {imp1.columns.tolist()}")

filtered = imp1[~((imp1["treat"] == 1) & (imp1["momwhite"] == 0))].reset_index(
    drop=True
)
race = cast(pd.DataFrame, filtered[["momwhite", "momblack", "momhisp"]])

for i in range(1, 11):
    npci = pd.read_csv(
        f"data/ihdp/npci/ihdp_npci_{i}.csv", names=NPCI_COLS, index_col=False
    )
    assert len(npci) == len(race), f"Length mismatch: {len(npci)} != {len(race)}"
    npci_with_race = pd.concat([npci, race], axis=1)
    npci_with_race.to_csv(f"data/ihdp/with_race/ihdp_with_race_{i}.csv", index=False)

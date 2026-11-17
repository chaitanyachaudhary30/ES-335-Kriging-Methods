"""Step 1.9: evaluate the non-spatial baselines on the outer test folds.

Run only after results/hypotheses.md has been committed: this is the first
script that reads test pH values.

Outputs
  results/baselines/<dataset>_baseline_metrics.csv     per fold
  results/baselines/<dataset>_baseline_summary.csv     mean and SD over folds
  results/baselines/<dataset>_baseline_predictions.csv per test point (predictions only, no observed pH)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.baselines import BASELINES  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import mae, rmse, skill  # noqa: E402

OUT = C.RESULTS / "baselines"
SPLIT_DIR = C.ROOT / "data" / "splits"


def run(name: str) -> None:
    df = build_features(pd.read_csv(C.DATA_PROCESSED / f"{name.lower()}_clean.csv"))
    sp = pd.read_csv(SPLIT_DIR / f"splits_{name.lower()}.csv")
    assert (sp["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()).all()
    z = df[C.TARGET].to_numpy()

    rows, preds = [], []
    for kind in ("spatial_fold", "random_fold"):
        split = kind.replace("_fold", "")
        for f in range(C.N_FOLDS):
            te = sp[kind].to_numpy() == f
            fold_rmse = {}
            for method, (feats, factory) in BASELINES.items():
                model = factory().fit(df.loc[~te, feats].to_numpy(), z[~te])
                p = model.predict(df.loc[te, feats].to_numpy())
                fold_rmse[method] = rmse(z[te], p)
                rows.append({"dataset": name, "split": split, "fold": f, "method": method,
                             "n_test": int(te.sum()), "rmse": fold_rmse[method], "mae": mae(z[te], p)})
                preds.append(pd.DataFrame({"id": df.loc[te, "id"].to_numpy(), "split": split, "fold": f,
                                           "method": method, "pred": np.round(p, 4)}))
            for r in rows[-len(BASELINES):]:
                r["skill"] = skill(r["rmse"], fold_rmse["Mean"])

    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / f"{name.lower()}_baseline_metrics.csv", index=False)
    pd.concat(preds).to_csv(OUT / f"{name.lower()}_baseline_predictions.csv", index=False)
    s = (m.groupby(["split", "method"])
         .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"),
              skill_mean=("skill", "mean"), skill_sd=("skill", "std"))
         .reset_index())
    s.to_csv(OUT / f"{name.lower()}_baseline_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        if (SPLIT_DIR / f"splits_{name.lower()}.csv").exists() and (C.DATA_PROCESSED / f"{name.lower()}_clean.csv").exists():
            run(name)
        else:
            print(f"{name}: data or splits missing, skipped")


if __name__ == "__main__":
    main()

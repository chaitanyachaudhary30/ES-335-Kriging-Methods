"""Step 2.2: choose model settings by inner cross-validation, separately for every outer fold.

Reads training pH values only (never the test fold's pH).

Outputs
  results/tuning/<dataset>_tuning_grid.csv      inner-CV RMSE of every setting
  results/tuning/<dataset>_tuned_settings.csv   chosen setting per outer fold and method
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.features import build_features  # noqa: E402
from src.tuning import tune_fold  # noqa: E402

OUT = C.RESULTS / "tuning"
SPLIT_DIR = C.ROOT / "data" / "splits"


def one_fold(df, sp, kind, f):
    te = sp[kind].to_numpy() == f
    train = df.loc[~te].reset_index(drop=True)
    train["inner_fold"] = sp.loc[~te, kind].to_numpy()
    chosen, grid = tune_fold(train, n_jobs=1)
    split = kind.replace("_fold", "")
    grid.insert(0, "fold", f)
    grid.insert(0, "split", split)
    return split, f, chosen, grid


def run(name: str) -> None:
    df = build_features(pd.read_csv(C.DATA_PROCESSED / f"{name.lower()}_clean.csv"))
    sp = pd.read_csv(SPLIT_DIR / f"splits_{name.lower()}.csv")
    assert (sp["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()).all()
    jobs = [(k, f) for k in ("spatial_fold", "random_fold") for f in range(C.N_FOLDS)]
    res = Parallel(n_jobs=min(len(jobs), os.cpu_count() or 1))(delayed(one_fold)(df, sp, k, f) for k, f in jobs)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([r[3] for r in res]).to_csv(OUT / f"{name.lower()}_tuning_grid.csv", index=False)
    rows = []
    for split, f, chosen, _ in res:
        for method, s in chosen.items():
            cv = s.pop("cv_rmse")
            rows.append({"dataset": name, "split": split, "fold": f, "method": method, "settings": json.dumps(s, sort_keys=True), "inner_cv_rmse": cv})
    t = pd.DataFrame(rows).sort_values(["split", "fold", "method"])
    t.to_csv(OUT / f"{name.lower()}_tuned_settings.csv", index=False)
    print(t.to_string(index=False))


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        if (SPLIT_DIR / f"splits_{name.lower()}.csv").exists() and (C.DATA_PROCESSED / f"{name.lower()}_clean.csv").exists():
            run(name)
        else:
            print(f"{name}: data or splits missing, skipped")


if __name__ == "__main__":
    main()

"""Step 2.3: evaluate all nine methods on the outer test folds of both splits.

Reads test pH values (after results/hypotheses.md and the tuning step are committed).
Settings come from results/tuning/<dataset>_tuned_settings.csv (07a_tune.py).

Outputs
  results/models/<dataset>_model_metrics.csv       per fold, same columns as the baseline metrics
  results/models/<dataset>_model_summary.csv       mean and SD over folds
  results/models/<dataset>_interval_coverage.csv   95% interval coverage and mean SD (OK, UK, GPR, RK)
  results/models/<dataset>_model_predictions.csv   per test point: id, split, fold, method, prediction, SD (no observed pH)
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import mae, rmse, skill  # noqa: E402
from src.models import KRIGING_METHODS, METHODS, build_model  # noqa: E402

OUT = C.RESULTS / "models"
SPLIT_DIR = C.ROOT / "data" / "splits"


def one_fold(name, df, sp, tuned, kind, f):
    split = kind.replace("_fold", "")
    te = sp[kind].to_numpy() == f
    train = df.loc[~te].reset_index(drop=True)
    train["inner_fold"] = sp.loc[~te, kind].to_numpy()
    test = df.loc[te].reset_index(drop=True)
    y = test[C.TARGET].to_numpy()
    rows, preds, cov = [], [], []
    for method in METHODS:
        s = tuned.get((split, f, method), {})
        p, sd = build_model(method, s, n_jobs=1).fit(train).predict(test)
        rows.append({"dataset": name, "split": split, "fold": f, "method": method,
                     "n_test": int(te.sum()), "rmse": rmse(y, p), "mae": mae(y, p)})
        preds.append(pd.DataFrame({"id": test["id"].to_numpy(), "split": split, "fold": f, "method": method,
                                   "pred": np.round(p, 4), "sd": np.round(sd, 4) if sd is not None else np.nan}))
        if sd is not None:
            cov.append({"dataset": name, "split": split, "fold": f, "method": method,
                        "coverage_95": float(np.mean(np.abs(y - p) <= 1.96 * sd)),
                        "mean_sd": float(sd.mean()), "rmse_over_mean_sd": rmse(y, p) / float(sd.mean())})
    base = next(r["rmse"] for r in rows if r["method"] == "Mean")
    for r in rows:
        r["skill"] = skill(r["rmse"], base)
    return rows, preds, cov


def run(name: str) -> None:
    df = build_features(pd.read_csv(C.DATA_PROCESSED / f"{name.lower()}_clean.csv"))
    sp = pd.read_csv(SPLIT_DIR / f"splits_{name.lower()}.csv")
    assert (sp["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()).all()
    t = pd.read_csv(C.RESULTS / "tuning" / f"{name.lower()}_tuned_settings.csv")
    tuned = {(r.split, int(r.fold), r.method): json.loads(r.settings) for r in t.itertuples()}

    jobs = [(k, f) for k in ("spatial_fold", "random_fold") for f in range(C.N_FOLDS)]
    res = Parallel(n_jobs=min(len(jobs), os.cpu_count() or 1))(delayed(one_fold)(name, df, sp, tuned, k, f) for k, f in jobs)

    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame([r for x in res for r in x[0]])
    m.to_csv(OUT / f"{name.lower()}_model_metrics.csv", index=False)
    pd.concat([p for x in res for p in x[1]]).to_csv(OUT / f"{name.lower()}_model_predictions.csv", index=False)
    cov = pd.DataFrame([c for x in res for c in x[2]])
    cov.to_csv(OUT / f"{name.lower()}_interval_coverage.csv", index=False)
    s = (m.groupby(["split", "method"])
         .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"),
              skill_mean=("skill", "mean"), skill_sd=("skill", "std"))
         .reset_index())
    s.to_csv(OUT / f"{name.lower()}_model_summary.csv", index=False)

    # The four non-spatial methods must reproduce the committed Phase 1 baseline metrics.
    b = pd.read_csv(C.RESULTS / "baselines" / f"{name.lower()}_baseline_metrics.csv")
    j = b.merge(m, on=["split", "fold", "method"], suffixes=("_b", "_m"))
    print(f"baseline reproduction: {len(j)} rows, max |rmse diff| = {np.abs(j['rmse_b'] - j['rmse_m']).max():.2e}")
    for split in ("spatial", "random"):
        print(f"\n{split} split")
        print(s[s.split == split].sort_values("skill_mean", ascending=False).round(3).to_string(index=False))
    print("\ninterval coverage (nominal 0.95)")
    print(cov.groupby(["split", "method"])[["coverage_95", "mean_sd", "rmse_over_mean_sd"]].mean().round(3).to_string())


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        if (SPLIT_DIR / f"splits_{name.lower()}.csv").exists() and (C.DATA_PROCESSED / f"{name.lower()}_clean.csv").exists():
            run(name)
        else:
            print(f"{name}: data or splits missing, skipped")


if __name__ == "__main__":
    main()

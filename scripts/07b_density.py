"""Step 2.4: GEMAS thinning experiment (deviation D1), spatial split.

For each outer test fold, the training fold is thinned to a fraction of its
points (nested subsets, ``src/splits.py: sparsity_subset``). The test fold stays
the same. Every method is refitted on the thinned data, with the discrete
settings (variogram model, neighbours, drift degree, nu) chosen on the full
training fold in step 07a; variograms, trends and kernels are refitted.
The a-priori predicted skill is recomputed on the thinned data with the same
formulas as step 04. Skill is relative to the mean of the thinned training set.

Outputs
  results/density/<dataset>_density_metrics.csv   per fold, fraction, seed, method
  results/density/<dataset>_density_summary.csv   mean over seeds and folds per fraction and method
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
from src.apriori import predicted_skill  # noqa: E402
from src.features import COORD_FEATURES, build_features  # noqa: E402
from src.metrics import rmse, skill  # noqa: E402
from src.models import METHODS, build_model  # noqa: E402
from src.spatial_stats import nn_distances  # noqa: E402
from src.splits import add_sparsity_keys, sparsity_subset  # noqa: E402

OUT = C.RESULTS / "density"
SPLIT_DIR = C.ROOT / "data" / "splits"


def one_cell(name, df, sp, tuned, f, frac, seed):
    te = sp["spatial_fold"].to_numpy() == f
    train_idx = np.flatnonzero(~te)
    keys = sp[f"u_seed{seed}"].to_numpy()
    sub = sparsity_subset(train_idx, keys, frac)
    train = df.iloc[sub].reset_index(drop=True)
    train["inner_fold"] = sp["spatial_fold"].to_numpy()[sub]
    test = df.loc[te].reset_index(drop=True)
    y = test[C.TARGET].to_numpy()
    pred_sk = predicted_skill(train, test[COORD_FEATURES].to_numpy())
    rows = []
    for method in METHODS:
        p, _ = build_model(method, tuned.get((f, method), {}), n_jobs=1).fit(train).predict(test)
        rows.append({"dataset": name, "fold": f, "fraction": frac, "seed": seed, "method": method,
                     "n_train": len(train), "train_nn_median_km": float(np.median(nn_distances(train[COORD_FEATURES].to_numpy()))),
                     "rmse": rmse(y, p), "pred_skill": pred_sk[method]})
    base = next(r["rmse"] for r in rows if r["method"] == "Mean")
    for r in rows:
        r["skill"] = skill(r["rmse"], base)
    return rows


def run(name: str) -> None:
    df = build_features(pd.read_csv(C.DATA_PROCESSED / f"{name.lower()}_clean.csv"))
    sp = add_sparsity_keys(pd.read_csv(SPLIT_DIR / f"splits_{name.lower()}.csv"))
    assert (sp["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()).all()
    t = pd.read_csv(C.RESULTS / "tuning" / f"{name.lower()}_tuned_settings.csv")
    t = t[t.split == "spatial"]
    tuned = {(int(r.fold), r.method): json.loads(r.settings) for r in t.itertuples()}

    # Full data is identical for every seed, so it is computed once (seed 0).
    jobs = [(f, 1.0, 0) for f in range(C.N_FOLDS)]
    jobs += [(f, frac, s) for f in range(C.N_FOLDS) for frac in C.SPARSITY_FRACTIONS if frac < 1.0 for s in C.SPARSITY_SEEDS]
    res = Parallel(n_jobs=min(len(jobs), os.cpu_count() or 1))(delayed(one_cell)(name, df, sp, tuned, *j) for j in jobs)
    m = pd.DataFrame([r for x in res for r in x])
    full = m[m.fraction == 1.0]
    extra = [full.assign(seed=s) for s in C.SPARSITY_SEEDS if s != 0]
    m = pd.concat([m] + extra).sort_values(["fold", "fraction", "seed", "method"], ascending=[True, False, True, True])

    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / f"{name.lower()}_density_metrics.csv", index=False)

    per_fold = m.groupby(["fraction", "method", "fold"], as_index=False).agg(
        skill=("skill", "mean"), pred_skill=("pred_skill", "mean"), n_train=("n_train", "mean"), nn=("train_nn_median_km", "mean"))
    s = per_fold.groupby(["fraction", "method"]).agg(
        n_train=("n_train", "mean"), train_nn_median_km=("nn", "mean"),
        skill_mean=("skill", "mean"), skill_sd=("skill", "std"),
        pred_skill_mean=("pred_skill", "mean"), pred_skill_sd=("pred_skill", "std")).reset_index()
    s.to_csv(OUT / f"{name.lower()}_density_summary.csv", index=False)

    # Check: full-density a-priori numbers equal the committed Phase 1 predictions.
    ref = pd.read_csv(C.RESULTS / "hypotheses" / f"{name.lower()}_predictions.csv")
    ref = ref[ref.split == "spatial"].rename(columns={"pred_skill": "ref"})
    j = ref.merge(full[["fold", "method", "pred_skill"]], on=["fold", "method"])
    print(f"a-priori check at 100% density: {len(j)} rows, max |diff| = {np.abs(j['ref'] - j['pred_skill']).max():.2e}")
    piv = s.pivot(index="method", columns="fraction", values="skill_mean").loc[METHODS]
    print("\nobserved skill by training fraction\n", piv.round(3).to_string())
    piv = s.pivot(index="method", columns="fraction", values="pred_skill_mean").loc[METHODS]
    print("\npredicted skill by training fraction\n", piv.round(3).to_string())


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        if (SPLIT_DIR / f"splits_{name.lower()}.csv").exists() and (C.DATA_PROCESSED / f"{name.lower()}_clean.csv").exists():
            run(name)
        else:
            print(f"{name}: data or splits missing, skipped")


if __name__ == "__main__":
    main()

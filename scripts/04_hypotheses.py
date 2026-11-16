"""Step 1.8: training-only statistics and a-priori error predictions per fold.

For every outer fold (spatial and random) this script uses ONLY the training
rows (pH values) and the test LOCATIONS. Test pH values are never read here.

It computes:
  * training statistics: n, nearest-neighbour spacing, variogram (nugget ratio,
    range), trend R^2, Moran's I
  * inner cross-validated MSE of the trend models (Mean, LinReg, RF), with
    inner folds = the remaining outer folds (spatial blocks stay intact)
  * predicted RMSE ratio vs the mean baseline for OK, GPR, UK and RK, from the
    kriging variance at the test locations (src/kriging_theory.py)

Outputs
  results/hypotheses/<dataset>_training_stats.csv
  results/hypotheses/<dataset>_predictions.csv
  docs/data/<dataset>_predictions.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.baselines import BASELINES  # noqa: E402
from src.features import COORD_FEATURES, build_features  # noqa: E402
from src.kriging_theory import ok_variance  # noqa: E402
from src.spatial_stats import empirical_variogram, fit_variogram, morans_i, nn_distances, trend_r2  # noqa: E402

OUT = C.RESULTS / "hypotheses"
SPLIT_DIR = C.ROOT / "data" / "splits"

# Which trend model each trend+residual method uses.
TREND_FOR = {"UK": "LinReg (coords)", "RK (linear)": "LinReg (coords + covariates)", "RK (RF)": "Random Forest"}


def inner_cv(train: pd.DataFrame, inner_fold: np.ndarray) -> tuple[dict, dict]:
    """Out-of-fold predictions on the training set for every baseline."""
    z = train[C.TARGET].to_numpy()
    mse, oof = {}, {}
    for name, (feats, factory) in BASELINES.items():
        pred = np.empty(len(train))
        for f in np.unique(inner_fold):
            tr, va = inner_fold != f, inner_fold == f
            model = factory().fit(train.loc[tr, feats].to_numpy(), z[tr])
            pred[va] = model.predict(train.loc[va, feats].to_numpy())
        oof[name] = pred
        mse[name] = float(np.mean((z - pred) ** 2))
    return mse, oof


def fit_vario(xy, z):
    c, g, n = empirical_variogram(xy, z)
    best, _ = fit_variogram(c, g, n)
    return best


def run(name: str) -> None:
    df = build_features(pd.read_csv(C.DATA_PROCESSED / f"{name.lower()}_clean.csv"))
    sp = pd.read_csv(SPLIT_DIR / f"splits_{name.lower()}.csv")
    assert (sp["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()).all(), "split file out of sync"

    stats_rows, pred_rows = [], []
    for kind in ("spatial_fold", "random_fold"):
        split = kind.replace("_fold", "")
        for f in range(C.N_FOLDS):
            te = sp[kind].to_numpy() == f
            train = df.loc[~te].reset_index(drop=True)
            inner = sp.loc[~te, kind].to_numpy()
            xy_tr = train[COORD_FEATURES].to_numpy()
            xy_te = df.loc[te, COORD_FEATURES].to_numpy()  # locations only
            z = train[C.TARGET].to_numpy()

            vf = fit_vario(xy_tr, z)
            mse, oof = inner_cv(train, inner)
            nn_test = nn_distances(xy_te, xy_tr)

            stats_rows.append(
                {
                    "dataset": name, "split": split, "fold": f, "n_train": len(train),
                    "train_nn_median_km": float(np.median(nn_distances(xy_tr))),
                    "test_to_train_nn_median_km": float(np.median(nn_test)),
                    "variogram_model": vf.model, "nugget": vf.nugget, "partial_sill": vf.partial_sill,
                    "nugget_ratio": vf.nugget_ratio, "range_km": vf.practical_range,
                    "trend_r2_coords": trend_r2(xy_tr, z),
                    "trend_r2_all": trend_r2(train[BASELINES["Random Forest"][0]].to_numpy(), z),
                    "morans_i": morans_i(xy_tr, z, permutations=99)["I"],
                    **{f"cv_mse[{k}]": v for k, v in mse.items()},
                }
            )

            base = mse["Mean"]
            ratio = {k: v / base for k, v in mse.items()}
            ok_ratio = float(ok_variance(vf, xy_tr, xy_te).mean() / vf.sill)
            ratio["OK"] = ok_ratio
            ratio["GPR"] = ok_ratio  # GPR = simple/ordinary kriging under the same covariance (theory E5)
            for method, trend in TREND_FOR.items():
                rv = fit_vario(xy_tr, z - oof[trend])
                ratio[method] = ratio[trend] * float(ok_variance(rv, xy_tr, xy_te).mean() / rv.sill)

            for method, r in ratio.items():
                pred_rows.append(
                    {
                        "dataset": name, "split": split, "fold": f, "method": method,
                        "pred_mse_ratio": r, "pred_rmse_ratio": float(np.sqrt(r)),
                        "pred_skill": float(1 - np.sqrt(r)),
                        "sqrt_eta_bound": float(np.sqrt(vf.nugget_ratio)),
                    }
                )
            print(f"{name} {split} fold {f}: eta={vf.nugget_ratio:.2f} range={vf.practical_range:.0f} km, "
                  f"pred skill OK={1-np.sqrt(ratio['OK']):.3f} UK={1-np.sqrt(ratio['UK']):.3f} "
                  f"RK(lin)={1-np.sqrt(ratio['RK (linear)']):.3f} RK(RF)={1-np.sqrt(ratio['RK (RF)']):.3f} "
                  f"RF={1-np.sqrt(ratio['Random Forest']):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    st, pr = pd.DataFrame(stats_rows), pd.DataFrame(pred_rows)
    st.to_csv(OUT / f"{name.lower()}_training_stats.csv", index=False)
    pr.to_csv(OUT / f"{name.lower()}_predictions.csv", index=False)

    agg = (
        pr.groupby(["split", "method"])
        .agg(pred_skill_mean=("pred_skill", "mean"), pred_skill_sd=("pred_skill", "std"),
             pred_rmse_ratio_mean=("pred_rmse_ratio", "mean"))
        .reset_index()
    )
    agg["pred_rank"] = agg.groupby("split")["pred_skill_mean"].rank(ascending=False, method="min").astype(int)
    agg.to_csv(OUT / f"{name.lower()}_predictions_summary.csv", index=False)
    (C.DOCS_DATA / f"{name.lower()}_predictions.json").write_text(
        json.dumps({"per_fold": pr.to_dict("records"), "summary": agg.to_dict("records"),
                    "training_stats": st.to_dict("records")})
    )
    print(agg.sort_values(["split", "pred_rank"]).round(3).to_string(index=False))


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        if (SPLIT_DIR / f"splits_{name.lower()}.csv").exists() and (C.DATA_PROCESSED / f"{name.lower()}_clean.csv").exists():
            run(name)
        else:
            print(f"{name}: data or splits missing, skipped")


if __name__ == "__main__":
    main()

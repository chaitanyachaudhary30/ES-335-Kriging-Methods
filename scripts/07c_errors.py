"""Step 2.5: error analysis. Aggregates only; no per-sample pH or residual is written.

Reads the model predictions of step 07 and the observed pH of the cleaned data.

Outputs (results/errors/)
  <ds>_error_by_distance_fold.csv   per split, fold, method and distance bin: n, RMSE, MAE (bins with n >= 5)
  <ds>_error_by_distance.csv        the same pooled over the 5 folds of a split
  <ds>_cell_residuals.csv           mean residual (observed - predicted) per 100 km cell with >= 5 test samples
  <ds>_error_by_country.csv         RMSE per country (spatial split, >= 10 test samples)
  <ds>_error_by_fold.csv            RMSE per fold and method
  <ds>_error_by_ph_range.csv        RMSE and bias for pH < 4.5, 4.5 to 7.5, > 7.5
  fig_*.png                         figures
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.features import COORD_FEATURES  # noqa: E402
from src.models import METHODS  # noqa: E402
from src.spatial_stats import nn_distances  # noqa: E402

OUT = C.RESULTS / "errors"
SPLIT_DIR = C.ROOT / "data" / "splits"
BINS = [0, 50, 100, 150, 250, np.inf]
BIN_LABELS = ["0-50", "50-100", "100-150", "150-250", ">250"]
CELL_KM = 100
MIN_CELL, MIN_COUNTRY, MIN_BIN = 5, 10, 5
PH_BINS = [(-np.inf, 4.5, "pH < 4.5"), (4.5, 7.5, "4.5 to 7.5"), (7.5, np.inf, "pH > 7.5")]


def build_table(name):
    df = pd.read_csv(C.DATA_PROCESSED / f"{name.lower()}_clean.csv")
    sp = pd.read_csv(SPLIT_DIR / f"splits_{name.lower()}.csv")
    pr = pd.read_csv(C.RESULTS / "models" / f"{name.lower()}_model_predictions.csv")
    assert (sp["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()).all()
    xy = df[COORD_FEATURES].to_numpy()
    dist = pd.concat([
        pd.DataFrame({"id": df["id"].to_numpy()[sp[k].to_numpy() == f], "split": k.replace("_fold", ""), "fold": f,
                      "dist_km": nn_distances(xy[sp[k].to_numpy() == f], xy[sp[k].to_numpy() != f])})
        for k in ("spatial_fold", "random_fold") for f in range(C.N_FOLDS)])
    t = pr.merge(dist, on=["id", "split", "fold"]).merge(df[["id", "country", "ph", "x_km", "y_km"]], on="id")
    t["resid"] = t["ph"] - t["pred"]
    t["bin"] = pd.cut(t["dist_km"], BINS, labels=BIN_LABELS, right=False)
    return t


def agg_error(g):
    return pd.Series({"n": len(g), "rmse": float(np.sqrt(np.mean(g["resid"] ** 2))), "mae": float(np.mean(np.abs(g["resid"])))})


def run(name: str) -> None:
    t = build_table(name)
    OUT.mkdir(parents=True, exist_ok=True)
    key = name.lower()

    bf = t.groupby(["split", "fold", "method", "bin"], observed=True).apply(agg_error, include_groups=False).reset_index()
    bf = bf[bf.n >= MIN_BIN]
    bf["n"] = bf["n"].astype(int)
    bf.to_csv(OUT / f"{key}_error_by_distance_fold.csv", index=False)
    bp = t.groupby(["split", "method", "bin"], observed=True).apply(agg_error, include_groups=False).reset_index()
    bp["n"] = bp["n"].astype(int)
    bp.to_csv(OUT / f"{key}_error_by_distance.csv", index=False)

    # mean residual per 100 km cell
    t["cx"], t["cy"] = (t["x_km"] // CELL_KM).astype(int), (t["y_km"] // CELL_KM).astype(int)
    cells = t.groupby(["split", "method", "cx", "cy"]).agg(n=("resid", "size"), mean_residual=("resid", "mean")).reset_index()
    cells = cells[cells.n >= MIN_CELL].copy()
    tr = Transformer.from_crs(C.CRS_LAEA, C.CRS_WGS84, always_xy=True)
    lon, lat = tr.transform(((cells.cx + 0.5) * CELL_KM * 1000).to_numpy(), ((cells.cy + 0.5) * CELL_KM * 1000).to_numpy())
    cells["lat"], cells["lon"] = np.round(lat, 1), np.round(lon, 1)
    cells["mean_residual"] = cells["mean_residual"].round(3)
    cells.to_csv(OUT / f"{key}_cell_residuals.csv", index=False)

    sp = t[t.split == "spatial"]
    ct = sp.groupby(["country", "method"]).apply(agg_error, include_groups=False).reset_index()
    ct = ct[ct.n >= MIN_COUNTRY]
    ct["n"] = ct["n"].astype(int)
    ct.to_csv(OUT / f"{key}_error_by_country.csv", index=False)

    fe = t.groupby(["split", "fold", "method"]).apply(agg_error, include_groups=False).reset_index()
    fe.to_csv(OUT / f"{key}_error_by_fold.csv", index=False)

    rows = []
    for split, g in t.groupby("split"):
        for lo, hi, label in PH_BINS:
            gg = g[(g.ph >= lo) & (g.ph < hi)] if np.isfinite(lo) else g[g.ph < hi]
            for method, m in gg.groupby("method"):
                rows.append({"split": split, "range": label, "method": method, "n": len(m),
                             "rmse": float(np.sqrt(np.mean(m.resid ** 2))), "bias": float(m.resid.mean())})
    pd.DataFrame(rows).to_csv(OUT / f"{key}_error_by_ph_range.csv", index=False)

    figures(key, t, bp, cells, ct)
    summarise(bp, ct, fe, pd.read_csv(OUT / f"{key}_error_by_ph_range.csv"))


def figures(key, t, bp, cells, ct):
    shown = ["Mean", "Random Forest", "OK", "UK", "RK (linear)", "GPR"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, split in zip(ax, ("spatial", "random")):
        for m in shown:
            d = bp[(bp.split == split) & (bp.method == m)]
            a.plot(d["bin"].astype(str), d["mae"], marker="o", label=m)
        a.set_title(f"{split} split"); a.set_xlabel("distance to nearest training sample (km)")
    ax[0].set_ylabel("mean absolute error (pH units)"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"fig_{key}_error_vs_distance.png", dpi=130); plt.close(fig)

    c = cells[(cells.split == "spatial") & (cells.method == "RK (linear)")]
    fig, a = plt.subplots(figsize=(6, 5))
    sc = a.scatter(c.lon, c.lat, c=c.mean_residual, cmap="RdBu_r", vmin=-1.5, vmax=1.5, s=90 + 8 * c.n)
    a.set_title("Mean residual per 100 km cell, RK (linear), spatial split\n(observed - predicted, cells with >= 5 test samples)", fontsize=9)
    fig.colorbar(sc, label="pH units"); a.set_xlabel("lon"); a.set_ylabel("lat")
    fig.tight_layout(); fig.savefig(OUT / f"fig_{key}_cell_residuals.png", dpi=130); plt.close(fig)

    p = ct[ct.method.isin(["Mean", "RK (linear)"])].pivot(index="country", columns="method", values="rmse").sort_values("RK (linear)")
    fig, a = plt.subplots(figsize=(8, 4.5))
    p.plot.bar(ax=a); a.set_ylabel("RMSE (pH units)"); a.set_title("RMSE by country, spatial split (countries with >= 10 test samples)")
    fig.tight_layout(); fig.savefig(OUT / f"fig_{key}_error_by_country.png", dpi=130); plt.close(fig)


def summarise(bp, ct, fe, ph):
    print("pooled MAE by distance bin, spatial split")
    print(bp[bp.split == "spatial"].pivot(index="method", columns="bin", values="mae").loc[METHODS].round(3).to_string())
    print("\nbin sizes (spatial):", bp[(bp.split == 'spatial') & (bp.method == 'Mean')].set_index('bin')['n'].to_dict())
    print("\nRMSE by fold, spatial split")
    print(fe[fe.split == "spatial"].pivot(index="method", columns="fold", values="rmse").loc[METHODS].round(3).to_string())
    print("\nworst countries, RK (linear), spatial split")
    print(ct[ct.method == "RK (linear)"].sort_values("rmse", ascending=False).head(6).round(3).to_string(index=False))
    print("\nby pH range, spatial split")
    print(ph[(ph.split == "spatial") & ph.method.isin(["Mean", "RK (linear)", "Random Forest"])].round(3).to_string(index=False))


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        if (SPLIT_DIR / f"splits_{name.lower()}.csv").exists() and (C.DATA_PROCESSED / f"{name.lower()}_clean.csv").exists():
            run(name)
        else:
            print(f"{name}: data or splits missing, skipped")


if __name__ == "__main__":
    main()

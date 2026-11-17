"""Step 1.5: descriptive spatial statistics for each cleaned survey.

These statistics use all cleaned samples and are descriptive only (EDA and the
block-size choice). The statistics used to state the hypotheses are recomputed
on training folds only in 04_hypotheses.py.

Outputs
  results/spatial_stats/<dataset>_summary.json
  results/spatial_stats/<dataset>_*.png
  docs/data/<dataset>_stats.json        (aggregates only, for the website)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.features import ALL_FEATURES, COORD_FEATURES, COVARIATE_FEATURES, build_features  # noqa: E402
from src.spatial_stats import (  # noqa: E402
    empirical_variogram,
    fit_variogram,
    morans_i,
    nn_distances,
    occupied_area_km2,
    trend_r2,
)

OUT = C.RESULTS / "spatial_stats"


def residuals(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    return z - LinearRegression().fit(X, z).predict(X)


def variogram_block(xy, z, label):
    c, g, n = empirical_variogram(xy, z)
    best, fits = fit_variogram(c, g, n)
    h = np.linspace(0, float(c.max()), 200)
    return {
        "label": label,
        "empirical": {"h_km": c.round(2).tolist(), "gamma": g.round(5).tolist(), "pairs": n.astype(int).tolist()},
        "best": best.model,
        "fits": [f.to_dict() for f in fits],
        "curves": {f.model: {"h_km": h.round(2).tolist(), "gamma": f.gamma(h).round(5).tolist()} for f in fits},
    }, best


def analyse(name: str, df: pd.DataFrame) -> dict:
    df = build_features(df)
    xy = df[COORD_FEATURES].to_numpy()
    z = df[C.TARGET].to_numpy()

    nn = nn_distances(xy)
    area = occupied_area_km2(xy)
    lam = len(df) / area
    quad = PolynomialFeatures(2, include_bias=False).fit_transform(xy)

    v_raw, best_raw = variogram_block(xy, z, "pH")
    v_ct, best_ct = variogram_block(xy, residuals(xy, z), "Residual after linear coordinate trend")
    v_cv, best_cv = variogram_block(xy, residuals(df[ALL_FEATURES].to_numpy(), z), "Residual after coordinate + covariate trend")

    summary = {
        "dataset": name,
        "n": int(len(df)),
        "countries": int(df["country"].nunique()),
        "occupied_area_km2": area,
        "density_per_1000km2": 1000 * lam,
        "nn_km": {"mean": float(nn.mean()), "median": float(np.median(nn)), "p90": float(np.percentile(nn, 90))},
        "nn_expected_csr_km": float(1 / (2 * np.sqrt(lam))),
        "ph": {
            "mean": float(z.mean()),
            "sd": float(z.std(ddof=1)),
            "var": float(z.var(ddof=1)),
            "skew": float(stats.skew(z)),
            "min": float(z.min()),
            "q25": float(np.percentile(z, 25)),
            "median": float(np.median(z)),
            "q75": float(np.percentile(z, 75)),
            "max": float(z.max()),
        },
        "trend_r2": {
            "coords_linear": trend_r2(xy, z),
            "coords_quadratic": trend_r2(quad, z),
            "covariates_only": trend_r2(df[COVARIATE_FEATURES].to_numpy(), z),
            "coords_plus_covariates": trend_r2(df[ALL_FEATURES].to_numpy(), z),
        },
        "covariate_spearman": {c: float(stats.spearmanr(df[c], z).statistic) for c in COVARIATE_FEATURES},
        "morans_i_ph": morans_i(xy, z),
        "variogram": {"ph": best_raw.to_dict(), "resid_coords": best_ct.to_dict(), "resid_covariates": best_cv.to_dict()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name.lower()}_summary.json").write_text(json.dumps(summary, indent=2))

    # Figures for the repository (the website redraws them with Plotly).
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    sc = ax[0].scatter(df["x_km"], df["y_km"], c=z, s=4, cmap="RdYlBu")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"{name}: topsoil pH (CaCl2), n={len(df)}")
    plt.colorbar(sc, ax=ax[0])
    ax[1].hist(z, bins=40, color="0.4")
    ax[1].set_title("pH distribution")
    for v, col in ((v_raw, "k"), (v_ct, "tab:blue"), (v_cv, "tab:green")):
        e = v["empirical"]
        ax[2].plot(e["h_km"], e["gamma"], "o", color=col, ms=4, label=v["label"])
        cur = v["curves"][v["best"]]
        ax[2].plot(cur["h_km"], cur["gamma"], "-", color=col)
    ax[2].set_xlabel("distance h (km)")
    ax[2].set_ylabel("semivariance")
    ax[2].legend(fontsize=7)
    ax[2].set_title("Empirical and fitted variograms")
    fig.tight_layout()
    fig.savefig(OUT / f"{name.lower()}_overview.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(nn, bins=40, color="0.4")
    ax.axvline(summary["nn_expected_csr_km"], color="r", ls="--", label="expected under random placement")
    ax.set_xlabel("distance to nearest other sample (km)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{name.lower()}_nn.png", dpi=130)
    plt.close(fig)

    web = {
        "summary": summary,
        "variograms": [v_raw, v_ct, v_cv],
        "nn_hist": np.histogram(nn, bins=np.arange(0, 205, 5))[0].tolist(),
        "nn_bin_edges": np.arange(0, 205, 5).tolist(),
    }
    C.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (C.DOCS_DATA / f"{name.lower()}_stats.json").write_text(json.dumps(web))
    return summary


def main() -> None:
    for name in ("GEMAS", "LUCAS"):
        path = C.DATA_PROCESSED / f"{name.lower()}_clean.csv"
        if not path.exists():
            print(f"{name}: {path.name} not found, skipped")
            continue
        s = analyse(name, pd.read_csv(path))
        v = s["variogram"]["ph"]
        print(
            f"{name}: n={s['n']}, density={s['density_per_1000km2']:.2f}/1000km2, "
            f"median NN={s['nn_km']['median']:.1f} km, Moran I={s['morans_i_ph']['I']:.3f}, "
            f"trend R2 (coords)={s['trend_r2']['coords_linear']:.3f}, (all)={s['trend_r2']['coords_plus_covariates']:.3f}, "
            f"variogram {v['model']}: nugget ratio={v['nugget_ratio']:.2f}, range={v['practical_range']:.0f} km"
        )


if __name__ == "__main__":
    main()

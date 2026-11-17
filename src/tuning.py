"""Hyperparameter choice by inner cross-validation, inside one training fold.

The inner folds are the labels of the other outer folds (as in
``scripts/04_hypotheses.py: inner_cv``), so in the spatial split the inner
validation sets are whole regions. Test pH values are never used here.

Grids (small on purpose)
  OK           variogram model {Exponential, Spherical, Matern} x neighbours {16, 32, 64}
  UK           drift degree {1, 2} x variogram model x neighbours
  RK (linear)  variogram model x neighbours
  RK (RF)      variogram model x neighbours
  GPR          Matern smoothness nu {0.5, 1.5, 2.5}

Nesting. For the trend + kriging methods the whole procedure is repeated inside
every inner validation fold. The trend model, its out-of-fold residuals (with the
remaining inner labels as nested folds) and the variogram are computed from the
inner training part alone, then the validation fold is predicted. Nothing from
the validation fold reaches the trend, the residuals or the variogram.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import COORD_FEATURES
from .models import GPR, VARIOGRAM_CHOICES, fit_vario, inner_labels, krige_local, oof_trend, trend_spec

NEIGHBOURS = [16, 32, 64]
NUS = [0.5, 1.5, 2.5]

# (method, trend model, UK degree)
TREND_METHODS = [
    ("OK", None, None),
    ("UK", "poly1", 1),
    ("UK", "poly2", 2),
    ("RK (linear)", "linreg_cov", None),
    ("RK (RF)", "rf", None),
]


def _grid_for_trend(train, inner, trend, method, degree, n_jobs):
    """Inner-CV RMSE of every (variogram model, k) for one trend model."""
    z = train[C.TARGET].to_numpy()
    xy = train[COORD_FEATURES].to_numpy()
    sq = {(m, k): 0.0 for m in VARIOGRAM_CHOICES for k in NEIGHBOURS}
    for f in np.unique(inner):
        va = inner == f
        tr = train.loc[~va].reset_index(drop=True)
        if trend is None:
            resid, base = z[~va], np.zeros(va.sum())
        else:
            feats, factory = trend_spec(trend, n_jobs)
            resid = z[~va] - oof_trend(tr, trend, inner[~va], n_jobs)
            base = factory().fit(tr[feats].to_numpy(), z[~va]).predict(train.loc[va, feats].to_numpy())
        for m in VARIOGRAM_CHOICES:
            fit = fit_vario(xy[~va], resid, m)
            for k in NEIGHBOURS:
                kr, _ = krige_local(fit, xy[~va], resid, xy[va], k)
                sq[(m, k)] += float(np.sum((z[va] - (base + kr)) ** 2))
    return [{"method": method, "degree": degree, "variogram": m, "k": k, "cv_rmse": float(np.sqrt(s / len(z)))} for (m, k), s in sq.items()]


def tune_fold(train: pd.DataFrame, n_jobs: int = 1) -> tuple[dict, pd.DataFrame]:
    """Return ({method: chosen settings}, grid table with the inner-CV RMSE of every setting)."""
    z = train[C.TARGET].to_numpy()
    inner = inner_labels(train)
    rows = []
    for method, trend, degree in TREND_METHODS:
        rows += _grid_for_trend(train, inner, trend, method, degree, n_jobs)

    for nu in NUS:
        sq = 0.0
        for f in np.unique(inner):
            va = inner == f
            p, _ = GPR(nu).fit(train[~va]).predict(train[va])
            sq += float(np.sum((z[va] - p) ** 2))
        rows.append({"method": "GPR", "nu": nu, "cv_rmse": float(np.sqrt(sq / len(z)))})

    grid = pd.DataFrame(rows)
    chosen = {}
    for method, g in grid.groupby("method"):
        best = g.loc[g["cv_rmse"].idxmin()]
        if method == "GPR":
            s = {"nu": float(best["nu"])}
        else:
            s = {"variogram": str(best["variogram"]), "k": int(best["k"])}
            if method == "UK":
                s["degree"] = int(best["degree"])
        chosen[method] = {**s, "cv_rmse": float(best["cv_rmse"])}
    return chosen, grid

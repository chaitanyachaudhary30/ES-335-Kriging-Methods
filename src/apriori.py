"""A-priori skill predictions for one training set (same formulas as ``scripts/04_hypotheses.py``).

Uses training pH values and the test *locations* only. For the density
experiment this is applied to thinned training sets. At full density it
reproduces ``results/hypotheses/gemas_predictions.csv`` (checked in
``scripts/07b_density.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import COORD_FEATURES
from .kriging_theory import ok_variance
from .models import inner_labels, oof_trend
from .spatial_stats import empirical_variogram, fit_variogram

BASELINE_TREND = {"Mean": "mean", "LinReg (coords)": "poly1", "LinReg (coords + covariates)": "linreg_cov", "Random Forest": "rf"}
TREND_FOR = {"UK": "LinReg (coords)", "RK (linear)": "LinReg (coords + covariates)", "RK (RF)": "Random Forest"}


def _vario(xy, v):
    c, g, n = empirical_variogram(xy, v)
    return fit_variogram(c, g, n)[0]


def predicted_skill(train: pd.DataFrame, xy_test: np.ndarray, n_jobs: int = 1) -> dict[str, float]:
    """Predicted skill (1 - sqrt(MSE ratio)) for the nine methods."""
    z = train[C.TARGET].to_numpy()
    xy = train[COORD_FEATURES].to_numpy()
    inner = inner_labels(train)
    oof = {name: oof_trend(train, t, inner, n_jobs) for name, t in BASELINE_TREND.items()}
    mse = {name: float(np.mean((z - p) ** 2)) for name, p in oof.items()}
    ratio = {name: v / mse["Mean"] for name, v in mse.items()}
    vf = _vario(xy, z)
    ok = float(ok_variance(vf, xy, xy_test).mean() / vf.sill)
    ratio["OK"] = ratio["GPR"] = ok
    for method, trend in TREND_FOR.items():
        rv = _vario(xy, z - oof[trend])
        ratio[method] = ratio[trend] * float(ok_variance(rv, xy, xy_test).mean() / rv.sill)
    return {k: float(1 - np.sqrt(v)) for k, v in ratio.items()}

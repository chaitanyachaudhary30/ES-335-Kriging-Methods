"""Phase 2 models with one interface: ``fit(train_df)`` then ``predict(test_df) -> (mean, sd or None)``.

Inputs are data frames that went through ``build_features``. Everything that is
learned (scaling, trends, variograms, kernel hyperparameters, forest) is fitted
on the training frame passed to ``fit`` and nothing else.

Methods
  OK            Ordinary Kriging on pH, local neighbourhood of k training points.
  UK            Kriging of the residuals of a polynomial trend in the coordinates.
  RK (linear)   Kriging of the residuals of LinReg (coords + covariates).
  RK (RF)       Kriging of the residuals of the Random Forest.
  GPR           Gaussian process, Matern kernel + white noise, inputs in km.

UK and RK share one class (``TrendKriging``). The variogram and the kriged
values are the *out-of-fold* trend residuals of the training data, so an
overfitted trend (the forest) cannot make the residuals look smaller than they
will be at new places. The inner folds come from the column ``inner_fold`` of
the training frame (the labels of the other outer folds); without it, spatial
clusters from k-means are used.

The standard deviation returned by the kriging models is that of a *new
measurement* (it includes the nugget), computed from the kriging system. For
UK and RK it describes the residual process only and ignores the uncertainty of
the trend.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config as C
from .features import ALL_FEATURES, COORD_FEATURES
from .kriging_theory import N_NEIGHBOURS
from .spatial_stats import MODELS as VARIO_MODELS
from .spatial_stats import VariogramFit, empirical_variogram, fit_variogram

METHODS = [
    "Mean", "LinReg (coords)", "LinReg (coords + covariates)", "Random Forest",
    "OK", "GPR", "UK", "RK (linear)", "RK (RF)",
]
KRIGING_METHODS = ["OK", "UK", "GPR", "RK (linear)", "RK (RF)"]

DEFAULTS = {
    "OK": {"variogram": "Exponential", "k": N_NEIGHBOURS},
    "UK": {"degree": 1, "variogram": "Exponential", "k": N_NEIGHBOURS},
    "RK (linear)": {"variogram": "Exponential", "k": N_NEIGHBOURS},
    "RK (RF)": {"variogram": "Exponential", "k": N_NEIGHBOURS},
    "GPR": {"nu": 1.5},
}


# ---------------------------------------------------------------- trends
def _rf(n_jobs: int = -1):
    # Same settings as src/baselines.py; n_jobs does not change the result.
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=n_jobs, random_state=C.SEED)


def trend_spec(name: str, n_jobs: int = -1):
    """Return (feature list, factory) for a trend model."""
    if name == "poly1":
        return COORD_FEATURES, lambda: make_pipeline(StandardScaler(), LinearRegression())
    if name == "poly2":
        return COORD_FEATURES, lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression())
    if name == "linreg_cov":
        return ALL_FEATURES, lambda: make_pipeline(StandardScaler(), LinearRegression())
    if name == "rf":
        return ALL_FEATURES, lambda: _rf(n_jobs)
    if name == "mean":
        return COORD_FEATURES, lambda: DummyRegressor(strategy="mean")
    raise ValueError(name)


def baseline_spec(method: str, n_jobs: int = -1):
    return {
        "Mean": trend_spec("mean"),
        "LinReg (coords)": trend_spec("poly1"),
        "LinReg (coords + covariates)": trend_spec("linreg_cov"),
        "Random Forest": trend_spec("rf", n_jobs),
    }[method]


def inner_labels(train: pd.DataFrame) -> np.ndarray:
    """Inner fold labels: the column ``inner_fold`` if present, else 5 spatial k-means clusters."""
    if "inner_fold" in train.columns:
        return train["inner_fold"].to_numpy()
    return KMeans(5, n_init=3, random_state=C.SEED).fit_predict(train[COORD_FEATURES].to_numpy())


def oof_trend(train: pd.DataFrame, trend: str, inner: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold predictions of a trend model on the training frame."""
    feats, factory = trend_spec(trend, n_jobs)
    X, z = train[feats].to_numpy(), train[C.TARGET].to_numpy()
    out = np.empty(len(train))
    for f in np.unique(inner):
        va = inner == f
        out[va] = factory().fit(X[~va], z[~va]).predict(X[va])
    return out


# ---------------------------------------------------------------- kriging
def fit_vario(xy: np.ndarray, v: np.ndarray, model: str = "Exponential") -> VariogramFit:
    """Fit one variogram model type; fall back to the best of the three if that fit fails."""
    c, g, n = empirical_variogram(xy, v)
    try:
        best, _ = fit_variogram(c, g, n, models=(model,))
        return best
    except ValueError:
        best, _ = fit_variogram(c, g, n)
        return best


def krige_local(fit: VariogramFit, xy_tr: np.ndarray, v_tr: np.ndarray, xy_te: np.ndarray,
                k: int = N_NEIGHBOURS, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with the k nearest training points of each target location.

    Solves [C 1; 1' 0][lambda; mu] = [c0; 1] for every target and returns the
    prediction lambda'v and the variance of a new measurement,
    C0 + C1 - lambda'c0 - mu (same system as ``kriging_theory.ok_variance``).
    """
    k = int(min(k, len(xy_tr)))
    d0, idx = cKDTree(xy_tr).query(xy_te, k=k)
    cm = fit.covmodel()
    c1, sill = fit.partial_sill, fit.sill
    mean = np.empty(len(xy_te))
    var = np.empty(len(xy_te))
    ar = np.arange(k)
    for s in range(0, len(xy_te), chunk):
        ii, dd = idx[s:s + chunk], d0[s:s + chunk]
        n = len(ii)
        P = xy_tr[ii]
        H = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        A = np.ones((n, k + 1, k + 1))
        A[:, :k, :k] = c1 * cm.correlation(H.ravel()).reshape(H.shape)
        A[:, ar, ar] = sill
        A[:, k, k] = 0.0
        b = np.ones((n, k + 1))
        b[:, :k] = c1 * cm.correlation(dd.ravel()).reshape(n, k)
        sol = np.linalg.solve(A, b[..., None])[..., 0]
        lam, mu = sol[:, :k], sol[:, k]
        mean[s:s + n] = np.einsum("nk,nk->n", lam, v_tr[ii])
        var[s:s + n] = sill - np.einsum("nk,nk->n", lam, b[:, :k]) - mu
    return mean, np.sqrt(np.maximum(var, 1e-9))


# ---------------------------------------------------------------- models
class BaselineModel:
    def __init__(self, method: str, n_jobs: int = -1):
        self.method = method
        self.feats, self.factory = baseline_spec(method, n_jobs)

    def fit(self, train: pd.DataFrame):
        self.model = self.factory().fit(train[self.feats].to_numpy(), train[C.TARGET].to_numpy())
        return self

    def predict(self, test: pd.DataFrame):
        return self.model.predict(test[self.feats].to_numpy()), None


class TrendKriging:
    """OK (``trend=None``), UK (``poly1``/``poly2``) and RK (``linreg_cov``/``rf``)."""

    def __init__(self, trend: str | None = None, variogram: str = "Exponential", k: int = N_NEIGHBOURS, n_jobs: int = -1):
        self.trend, self.variogram, self.k, self.n_jobs = trend, variogram, k, n_jobs

    def fit(self, train: pd.DataFrame):
        self.xy = train[COORD_FEATURES].to_numpy()
        z = train[C.TARGET].to_numpy()
        if self.trend is None:
            self.resid = z
        else:
            self.feats, factory = trend_spec(self.trend, self.n_jobs)
            self.trend_model = factory().fit(train[self.feats].to_numpy(), z)
            self.resid = z - oof_trend(train, self.trend, inner_labels(train), self.n_jobs)
        self.vario = fit_vario(self.xy, self.resid, self.variogram)
        return self

    def predict(self, test: pd.DataFrame):
        kr, sd = krige_local(self.vario, self.xy, self.resid, test[COORD_FEATURES].to_numpy(), self.k)
        base = 0.0 if self.trend is None else self.trend_model.predict(test[self.feats].to_numpy())
        return base + kr, sd


class GPR:
    """Exact GP on coordinates in km: constant * Matern(nu) + white noise, targets standardised."""

    def __init__(self, nu: float = 1.5):
        self.nu = nu

    def fit(self, train: pd.DataFrame):
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=300.0, length_scale_bounds=(10.0, 5000.0), nu=self.nu) \
            + WhiteKernel(0.3, (1e-3, 5.0))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0, random_state=C.SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(train[COORD_FEATURES].to_numpy(), train[C.TARGET].to_numpy())
        return self

    def predict(self, test: pd.DataFrame):
        m, sd = self.gp.predict(test[COORD_FEATURES].to_numpy(), return_std=True)
        return m, sd


def build_model(method: str, settings: dict | None = None, n_jobs: int = -1):
    """Create an unfitted model. ``settings`` override ``DEFAULTS[method]``."""
    s = {**DEFAULTS.get(method, {}), **(settings or {})}
    if method == "OK":
        return TrendKriging(None, s["variogram"], s["k"], n_jobs)
    if method == "UK":
        return TrendKriging(f"poly{int(s['degree'])}", s["variogram"], s["k"], n_jobs)
    if method == "RK (linear)":
        return TrendKriging("linreg_cov", s["variogram"], s["k"], n_jobs)
    if method == "RK (RF)":
        return TrendKriging("rf", s["variogram"], s["k"], n_jobs)
    if method == "GPR":
        return GPR(s["nu"])
    return BaselineModel(method, n_jobs)


VARIOGRAM_CHOICES = list(VARIO_MODELS)

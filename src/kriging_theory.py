"""Model-based error predictions made *before* any test target is seen.

Ordinary Kriging gives, for each prediction location s0, the expected squared
error sigma^2_OK(s0) implied by the fitted variogram and the positions of the
training points. It needs the test *locations* (known at prediction time) but
not the test pH values. Averaging sigma^2_OK over the test locations and
dividing by the sill gives a predicted MSE ratio against the mean baseline:

    predicted MSE_OK / MSE_mean  ~  mean_s0 sigma^2_OK(s0) / (C0 + C1)

For trend + residual methods (UK, RK) the trend part is measured by inner
spatial cross-validation on the training data, and the residual part by the
same kriging-variance ratio on the residual variogram (see docs/theory.md).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .spatial_stats import VariogramFit

N_NEIGHBOURS = 64  # local kriging neighbourhood, shared with the Phase 2 models


def _cov(fit: VariogramFit, h: np.ndarray) -> np.ndarray:
    """Covariance of the measured process: C1 r(h) for h > 0, C0 + C1 at h = 0."""
    c = fit.partial_sill * fit.correlation(h)
    return np.where(h == 0, fit.sill, c)


def ok_variance(fit: VariogramFit, xy_train: np.ndarray, xy_test: np.ndarray, k: int = N_NEIGHBOURS) -> np.ndarray:
    """Ordinary kriging variance of a new *measurement* at each test location.

    Solves, per test point, the OK system with its k nearest training points:
        [C  1] [lambda]   [c0]
        [1' 0] [  mu  ] = [ 1]
    and returns sigma^2 = C(0) - lambda' c0 - mu, where C(0) = C0 + C1 because a
    new measurement carries its own nugget (measurement error + micro-scale variation).
    """
    k = min(k, len(xy_train))
    tree = cKDTree(xy_train)
    d0, idx = tree.query(xy_test, k=k)
    out = np.empty(len(xy_test))
    A = np.ones((k + 1, k + 1))
    A[k, k] = 0.0
    b = np.ones(k + 1)
    for i in range(len(xy_test)):
        pts = xy_train[idx[i]]
        H = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        A[:k, :k] = _cov(fit, H)
        b[:k] = fit.partial_sill * fit.correlation(d0[i])
        sol = np.linalg.solve(A, b)
        lam, mu = sol[:k], sol[k]
        out[i] = fit.sill - lam @ b[:k] - mu
    return out


def single_neighbour_ratio(nugget_ratio: float, r: np.ndarray) -> np.ndarray:
    """Simple-kriging MSE ratio with one neighbour at correlation r: 1 - (1 - eta)^2 r^2."""
    return 1.0 - (1.0 - nugget_ratio) ** 2 * np.asarray(r) ** 2

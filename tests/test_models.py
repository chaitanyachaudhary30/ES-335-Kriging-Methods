"""Checks on synthetic data (no survey data needed). Run: python tests/test_models.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kriging_theory import ok_variance  # noqa: E402
from src.models import METHODS, build_model, krige_local  # noqa: E402
from src.spatial_stats import VariogramFit  # noqa: E402


def synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1000, (n, 2))
    z = 5 + 0.002 * xy[:, 0] + np.sin(xy[:, 1] / 150) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x_km": xy[:, 0], "y_km": xy[:, 1], "ph": z})
    for c, s in [("elev", 200), ("bio1", 3), ("bio4", 100), ("log_bio12", 0.3), ("bio15", 8)]:
        df[c] = rng.normal(0, s, n) + (0.001 * z if c == "bio1" else 0)
    return df


def test_krige_matches_phase1_solver():
    fit = VariogramFit("Exponential", 0.3, 0.7, 200.0, None, 600.0, 1.0)
    rng = np.random.default_rng(1)
    xy, xt = rng.uniform(0, 1000, (200, 2)), rng.uniform(0, 1000, (20, 2))
    _, sd = krige_local(fit, xy, rng.normal(size=200), xt, k=32)
    ref = np.sqrt(ok_variance(fit, xy, xt, k=32))
    assert np.allclose(sd, ref, atol=1e-6), np.abs(sd - ref).max()


def test_exact_at_training_point_without_nugget():
    fit = VariogramFit("Exponential", 0.0, 1.0, 200.0, None, 600.0, 1.0)
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 1000, (100, 2))
    v = rng.normal(size=100)
    m, _ = krige_local(fit, xy, v, xy[:5] + 1e-6, k=16)
    assert np.allclose(m, v[:5], atol=1e-3)


def test_all_methods_fit_predict():
    tr, te = synthetic(300, 0), synthetic(40, 5)
    tr["inner_fold"] = np.arange(len(tr)) % 4
    for m in METHODS:
        p, sd = build_model(m, n_jobs=1).fit(tr).predict(te)
        assert p.shape == (40,) and np.isfinite(p).all(), m
        assert sd is None or (sd.shape == (40,) and (sd > 0).all()), m
    print("all methods OK")


if __name__ == "__main__":
    test_krige_matches_phase1_solver()
    test_exact_at_training_point_without_nugget()
    test_all_methods_fit_predict()
    print("passed")

"""Non-spatial baselines: training mean, linear regression, random forest.

Every model is a scikit-learn pipeline, so scaling is learned on the training
fold only. Random Forest settings are fixed here (no tuning in Phase 1);
Phase 2 may tune them with inner spatial CV.
"""
from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import ALL_FEATURES, COORD_FEATURES

BASELINES = {
    # name: (feature list, factory)
    "Mean": (COORD_FEATURES, lambda: DummyRegressor(strategy="mean")),
    "LinReg (coords)": (COORD_FEATURES, lambda: make_pipeline(StandardScaler(), LinearRegression())),
    "LinReg (coords + covariates)": (ALL_FEATURES, lambda: make_pipeline(StandardScaler(), LinearRegression())),
    "Random Forest": (
        ALL_FEATURES,
        lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=C.SEED
        ),
    ),
}

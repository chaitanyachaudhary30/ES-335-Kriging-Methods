"""Feature sets used by the baselines, Universal Kriging and Regression Kriging.

Only inputs that are known at an unsampled location are allowed: projected
coordinates and gridded WorldClim covariates. Lab-measured soil properties
(organic carbon, CaCO3, CEC, ...) are excluded because they would not be
available where we want to predict.
"""
import numpy as np
import pandas as pd

COORD_FEATURES = ["x_km", "y_km"]
COVARIATE_FEATURES = ["elev", "bio1", "bio4", "log_bio12", "bio15"]
ALL_FEATURES = COORD_FEATURES + COVARIATE_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Precipitation is log-transformed because it is right-skewed
    and its effect on leaching (and hence pH) is closer to multiplicative."""
    out = df.copy()
    out["log_bio12"] = np.log1p(out["bio12"])
    return out

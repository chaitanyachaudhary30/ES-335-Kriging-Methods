"""Evaluation metrics."""
import numpy as np


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def skill(rmse_model: float, rmse_mean: float) -> float:
    """Fractional RMSE improvement over the training-mean baseline (0 = no gain, 1 = perfect)."""
    return 1.0 - rmse_model / rmse_mean

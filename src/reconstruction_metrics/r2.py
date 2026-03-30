"""Coefficient of determination (sklearn) on the missing-value mask."""

import numpy as np
from sklearn.metrics import r2_score

from ._registry import EPS, register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or float(np.var(y_true)) < EPS:
        return float("nan")
    return float(r2_score(y_true, y_pred))


register_builtin_metric("r2", "R² (coefficient of determination)", False, compute)

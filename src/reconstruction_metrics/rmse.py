"""Root mean square error on the missing-value mask."""

import numpy as np

from ._registry import register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


register_builtin_metric("rmse", "RMSE", True, compute)

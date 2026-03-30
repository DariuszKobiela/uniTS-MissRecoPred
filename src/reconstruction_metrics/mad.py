"""Mean absolute difference (same as MAE on the missing-value mask)."""

import numpy as np

from ._registry import register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


register_builtin_metric("mad", "MAD", True, compute)

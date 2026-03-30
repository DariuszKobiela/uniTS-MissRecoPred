"""Symmetric mean absolute percentage error (%)."""

import numpy as np

from ._registry import EPS, register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray, *, train=None) -> float:
    del train
    denom = np.abs(y_true) + np.abs(y_pred) + EPS
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))


register_builtin_metric(
    "smape", "SMAPE (%)", True, compute, needs_train=False, value_is_percent=True
)

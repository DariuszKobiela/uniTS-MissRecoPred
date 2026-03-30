"""Mean absolute percentage error (%); undefined when all actuals are zero."""

import numpy as np

from ._registry import register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray, *, train=None) -> float:
    del train
    nz = y_true != 0
    if not np.any(nz):
        return float("nan")
    e = y_true - y_pred
    return float(np.mean(np.abs(e[nz]) / np.abs(y_true[nz])) * 100.0)


register_builtin_metric(
    "mape", "MAPE (%)", True, compute, needs_train=False, value_is_percent=True
)

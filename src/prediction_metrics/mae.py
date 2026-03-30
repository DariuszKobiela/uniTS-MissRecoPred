"""Mean absolute error on the test horizon."""

import numpy as np

from ._registry import register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray, *, train=None) -> float:
    del train
    return float(np.mean(np.abs(y_true - y_pred)))


register_builtin_metric("mae", "MAE", True, compute, value_is_percent=False)

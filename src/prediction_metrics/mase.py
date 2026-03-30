"""
Mean absolute scaled error. Scale = in-sample MAE of naive one-step forecasts on ``train``
(mean absolute first difference). Numerator = MAE on test (y_true vs y_pred).
Train series should match ``data/2_splitted_data/train/{dataset}.csv`` (first value column).
"""

import numpy as np

from ._registry import EPS, register_builtin_metric


def compute(y_true: np.ndarray, y_pred: np.ndarray, *, train=None) -> float:
    if train is None:
        return float("nan")
    tr = np.asarray(train, dtype=np.float64).ravel()
    tr = tr[~np.isnan(tr)]
    if tr.size < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(tr)))
    if scale < EPS:
        return float("nan")
    mae_test = np.mean(np.abs(y_true - y_pred))
    return float(mae_test / scale)


register_builtin_metric(
    "mase", "MASE", True, compute, needs_train=True, value_is_percent=False
)

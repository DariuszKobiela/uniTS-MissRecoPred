"""
Prediction error metrics for test-set forecasts.

Add ``your_metric.py`` with ``compute(y_true, y_pred, *, train=None)`` and
``register_builtin_metric(...)``, then import the module below.
"""

from __future__ import annotations

from . import mae as _mae  # noqa: F401
from . import mape as _mape  # noqa: F401
from . import mase as _mase  # noqa: F401
from . import rmse as _rmse  # noqa: F401
from . import smape as _smape  # noqa: F401

from ._registry import (
    EPS,
    PredictionMetricSpec,
    compute_prediction_metrics,
    get_metric_spec,
    infer_lower_is_better,
    list_metric_specs_ordered,
    list_primary_metric_keys,
    register_builtin_metric,
    register_prediction_metric,
)

__all__ = [
    "EPS",
    "PredictionMetricSpec",
    "compute_prediction_metrics",
    "get_metric_spec",
    "infer_lower_is_better",
    "list_metric_specs_ordered",
    "list_primary_metric_keys",
    "register_builtin_metric",
    "register_prediction_metric",
]

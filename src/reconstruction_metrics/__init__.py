"""
Reconstruction error metrics (missing positions only).

Add a new metric: create ``your_metric.py`` in this package with a ``compute(y_true, y_pred)``
and call ``register_builtin_metric(...)``, then import the module below (alphabetical order optional).
Custom / plugin metrics can use ``register_reconstruction_metric`` from code outside this package.

Built-in modules: mad, mae, rmse, r2, smape.
"""

from __future__ import annotations

# Side-effect: register built-ins in stable order
from . import mad as _mad  # noqa: F401
from . import mae as _mae  # noqa: F401
from . import rmse as _rmse  # noqa: F401
from . import r2 as _r2  # noqa: F401
from . import smape as _smape  # noqa: F401

from ._registry import (
    EPS,
    ReconstructionMetricSpec,
    align_missing_values_series,
    compute_metrics_from_series,
    compute_reconstruction_metrics,
    get_metric_spec,
    infer_lower_is_better,
    list_metric_specs_ordered,
    list_primary_metric_keys,
    metric_keys_for_csv,
    optimization_loss,
    register_builtin_metric,
    register_reconstruction_metric,
)

__all__ = [
    "EPS",
    "ReconstructionMetricSpec",
    "align_missing_values_series",
    "compute_metrics_from_series",
    "compute_reconstruction_metrics",
    "get_metric_spec",
    "infer_lower_is_better",
    "list_metric_specs_ordered",
    "list_primary_metric_keys",
    "metric_keys_for_csv",
    "optimization_loss",
    "register_builtin_metric",
    "register_reconstruction_metric",
]

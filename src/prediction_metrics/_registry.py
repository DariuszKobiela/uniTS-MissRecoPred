"""
Registry for prediction error metrics (test horizon: y_true vs y_pred).

MASE uses training series for the naive scale; other metrics ignore ``train``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

EPS = 1e-12

# Columns written by script 9 outside primary metrics
_RESERVED = frozenset({"max_error", "min_error", "std_error", "n_samples"})


@dataclass(frozen=True)
class PredictionMetricSpec:
    """Metadata for one primary prediction metric (CSV / Streamlit)."""

    key: str
    label: str
    lower_is_better: bool
    value_is_percent: bool = False


MetricCompute = Callable[..., float]  # (y_true, y_pred, *, train=None) -> float

_PRIMARY_ORDER: List[str] = []
_SPECS: Dict[str, PredictionMetricSpec] = {}
_NEEDS_TRAIN: Dict[str, bool] = {}
_BUILTIN_COMPUTE: Dict[str, MetricCompute] = {}
_USER_METRICS: Dict[str, tuple] = {}  # key -> (spec, fn, needs_train)


def register_builtin_metric(
    key: str,
    label: str,
    lower_is_better: bool,
    compute: MetricCompute,
    *,
    needs_train: bool = False,
    value_is_percent: bool = False,
) -> None:
    key = key.strip().lower()
    if key in _RESERVED:
        raise ValueError(f"Key '{key}' is reserved")
    if key in _SPECS:
        return
    _SPECS[key] = PredictionMetricSpec(
        key=key, label=label, lower_is_better=lower_is_better, value_is_percent=value_is_percent
    )
    _PRIMARY_ORDER.append(key)
    _NEEDS_TRAIN[key] = needs_train
    _BUILTIN_COMPUTE[key] = compute


def register_prediction_metric(
    key: str,
    label: str,
    lower_is_better: bool,
    compute: MetricCompute,
    *,
    needs_train: bool = False,
    value_is_percent: bool = False,
) -> None:
    key = key.strip().lower()
    if not key:
        raise ValueError("Metric key must be non-empty")
    if key in _RESERVED:
        raise ValueError(f"Key '{key}' is reserved")
    if key in _BUILTIN_COMPUTE:
        raise ValueError(f"Cannot override built-in metric '{key}'")
    spec = PredictionMetricSpec(
        key=key, label=label, lower_is_better=lower_is_better, value_is_percent=value_is_percent
    )
    _USER_METRICS[key] = (spec, compute, needs_train)
    _SPECS[key] = spec
    _NEEDS_TRAIN[key] = needs_train
    if key not in _PRIMARY_ORDER:
        _PRIMARY_ORDER.append(key)


def get_metric_spec(key: str) -> PredictionMetricSpec:
    k = key.strip().lower()
    if k not in _SPECS:
        raise KeyError(f"Unknown prediction metric: {key!r}. Known: {list(_SPECS.keys())}")
    return _SPECS[k]


def list_primary_metric_keys() -> List[str]:
    return list(_PRIMARY_ORDER)


def list_metric_specs_ordered() -> List[PredictionMetricSpec]:
    return [_SPECS[k] for k in _PRIMARY_ORDER if k in _SPECS]


def infer_lower_is_better(metric_key: str) -> bool:
    return get_metric_spec(metric_key).lower_is_better


def _resolve_keys(metric_keys: Optional[List[str]]) -> List[str]:
    if not metric_keys:
        return list(_PRIMARY_ORDER)
    out: List[str] = []
    for k in metric_keys:
        kk = str(k).strip().lower()
        if kk not in _SPECS:
            raise KeyError(f"Unknown prediction metric: {kk!r}")
        if kk not in out:
            out.append(kk)
    return out


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train: Optional[np.ndarray] = None,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute selected primary metrics on aligned finite arrays.

    Args:
        y_true, y_pred: 1-D float arrays, same length.
        train: Optional training series (1-D) for MASE scaling.
        metric_keys: Subset of registered keys; None or [] means all registered.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("No values to score")

    keys = _resolve_keys(metric_keys)
    out: Dict[str, float] = {}

    for key in keys:
        needs = _NEEDS_TRAIN.get(key, False)
        if needs:
            if train is None:
                out[key] = float("nan")
                continue
            tr = np.asarray(train, dtype=np.float64).ravel()
            tr = tr[~np.isnan(tr)]
            if tr.size < 2:
                out[key] = float("nan")
                continue
        else:
            tr = train  # may be None; compute ignores

        if key in _USER_METRICS:
            _, fn, _ = _USER_METRICS[key]
            out[key] = float(fn(yt, yp, train=tr if needs else None))
        elif key in _BUILTIN_COMPUTE:
            fn = _BUILTIN_COMPUTE[key]
            out[key] = float(fn(yt, yp, train=tr if needs else None))
        else:
            raise RuntimeError(f"No compute function for {key!r}")

    return out

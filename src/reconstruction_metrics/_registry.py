"""
Registry and aggregation for reconstruction error metrics (missing positions only).

Individual metrics live in sibling modules (e.g. mad.py, smape.py); each registers
via register_builtin_metric() at import time. Custom metrics use register_reconstruction_metric().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class ReconstructionMetricSpec:
    """Metadata for one primary reconstruction metric (CSV / UI / optimization)."""

    key: str
    label: str
    lower_is_better: bool


_PRIMARY_ORDER: List[str] = []
_SPECS: Dict[str, ReconstructionMetricSpec] = {}
_USER_METRICS: Dict[str, Tuple[ReconstructionMetricSpec, Callable[[np.ndarray, np.ndarray], float]]] = {}
_BUILTIN_COMPUTE: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}

_RESERVED = frozenset({"max_diff", "min_diff", "std_diff", "n_missing", "n_total"})


def register_builtin_metric(
    key: str,
    label: str,
    lower_is_better: bool,
    compute: Callable[[np.ndarray, np.ndarray], float],
) -> None:
    """Register a built-in metric (called from each metric module once)."""
    key = key.strip().lower()
    if key in _RESERVED:
        raise ValueError(f"Key '{key}' is reserved for auxiliary fields")
    if key in _SPECS:
        return
    _SPECS[key] = ReconstructionMetricSpec(key=key, label=label, lower_is_better=lower_is_better)
    _PRIMARY_ORDER.append(key)
    _BUILTIN_COMPUTE[key] = compute


def register_reconstruction_metric(
    key: str,
    label: str,
    lower_is_better: bool,
    compute: Callable[[np.ndarray, np.ndarray], float],
) -> None:
    """
    Register a custom metric. ``compute`` receives 1-D float arrays y_true, y_pred.

    Raises:
        ValueError: if key is reserved or collides with a built-in metric.
    """
    key = key.strip().lower()
    if not key:
        raise ValueError("Metric key must be non-empty")
    if key in _RESERVED:
        raise ValueError(f"Key '{key}' is reserved for auxiliary fields")
    if key in _BUILTIN_COMPUTE:
        raise ValueError(f"Cannot override built-in metric '{key}'")
    spec = ReconstructionMetricSpec(key=key, label=label, lower_is_better=lower_is_better)
    _USER_METRICS[key] = (spec, compute)
    _SPECS[key] = spec
    if key not in _PRIMARY_ORDER:
        _PRIMARY_ORDER.append(key)


def get_metric_spec(key: str) -> ReconstructionMetricSpec:
    k = key.strip().lower()
    if k not in _SPECS:
        raise KeyError(f"Unknown reconstruction metric: {key!r}. Known: {list(_SPECS.keys())}")
    return _SPECS[k]


def list_primary_metric_keys() -> List[str]:
    return list(_PRIMARY_ORDER)


def list_metric_specs_ordered() -> List[ReconstructionMetricSpec]:
    return [_SPECS[k] for k in _PRIMARY_ORDER if k in _SPECS]


def metric_keys_for_csv() -> List[str]:
    return list_primary_metric_keys()


def _resolve_metric_keys(metric_keys: Optional[List[str]]) -> List[str]:
    """Resolve which primary keys to compute; ``None`` or empty list → all registered."""
    if not metric_keys:
        return list(_PRIMARY_ORDER)
    out: List[str] = []
    for k in metric_keys:
        kk = str(k).strip().lower()
        if kk not in _SPECS:
            raise KeyError(
                f"Unknown reconstruction metric: {kk!r}. Known: {list(_SPECS.keys())}"
            )
        if kk not in out:
            out.append(kk)
    return out


def compute_reconstruction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute selected registered primary metrics plus auxiliary diff stats.

    Args:
        y_true, y_pred: 1-D arrays, same length, finite values (missing positions only).
        metric_keys: Subset of registered keys; ``None`` or ``[]`` → all primaries.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("No values to score")

    diff = yt - yp
    adiff = np.abs(diff)

    keys = _resolve_metric_keys(metric_keys)
    out: Dict[str, float] = {}
    for key in keys:
        if key in _USER_METRICS:
            _, fn = _USER_METRICS[key]
            out[key] = fn(yt, yp)
        elif key in _BUILTIN_COMPUTE:
            out[key] = _BUILTIN_COMPUTE[key](yt, yp)
        else:
            raise RuntimeError(f"Metric {key!r} has no compute function")

    out["max_diff"] = float(np.max(adiff))
    out["min_diff"] = float(np.min(adiff))
    out["std_diff"] = float(np.std(adiff, ddof=0))
    out["n_missing"] = float(len(yt))
    return out


def align_missing_values_series(
    source: pd.Series,
    degraded: pd.Series,
    reconstructed: pd.Series,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract (y_true, y_pred) where ``degraded`` is NaN; drop invalid pairs."""
    common_index = source.index.intersection(degraded.index).intersection(reconstructed.index)
    if len(common_index) == 0:
        return None

    source_vals = source.loc[common_index]
    degraded_vals = degraded.loc[common_index]
    recon_vals = reconstructed.loc[common_index]

    missing_mask = degraded_vals.isna()
    if not missing_mask.any():
        return None

    s = pd.to_numeric(source_vals[missing_mask], errors="coerce")
    r = pd.to_numeric(recon_vals[missing_mask], errors="coerce")
    valid = ~(s.isna() | r.isna())
    if not valid.any():
        return None

    yt = s[valid].to_numpy(dtype=np.float64)
    yp = r[valid].to_numpy(dtype=np.float64)
    return yt, yp


def compute_metrics_from_series(
    source: pd.Series,
    degraded: pd.Series,
    reconstructed: pd.Series,
) -> Optional[Dict[str, float]]:
    """Full metric dict for optimization, or None if no valid missing points."""
    aligned = align_missing_values_series(source, degraded, reconstructed)
    if aligned is None:
        return None
    return compute_reconstruction_metrics(aligned[0], aligned[1])


def optimization_loss(metric_value: float, lower_is_better: bool) -> float:
    """Scalar for Optuna ``minimize`` (higher-is-better metrics are negated)."""
    if metric_value is None or (isinstance(metric_value, float) and np.isnan(metric_value)):
        return float("inf")
    v = float(metric_value)
    return v if lower_is_better else -v


def infer_lower_is_better(metric_key: str) -> bool:
    return get_metric_spec(metric_key).lower_is_better

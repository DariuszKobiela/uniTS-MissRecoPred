"""Utilities for expanding-history rolling-origin forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from prediction_metrics import compute_prediction_metrics


@dataclass(frozen=True)
class RollingOrigin:
    """One forecast origin expressed as an offset inside the fixed test split."""

    number: int
    test_offset: int
    horizon: int

    @property
    def test_stop(self) -> int:
        return self.test_offset + self.horizon


def plan_rolling_origins(
    test_length: int,
    horizon: int,
    n_origins: int = 5,
) -> list[RollingOrigin]:
    """Spread up to n_origins unique origins across a fixed test split."""
    if test_length < 1:
        raise ValueError("test_length must be positive")
    if horizon < 1 or horizon > test_length:
        raise ValueError("horizon must be in [1, test_length]")
    if n_origins < 1:
        raise ValueError("n_origins must be positive")

    max_offset = test_length - horizon
    count = min(n_origins, max_offset + 1)
    if count == 1:
        offsets = [0]
    else:
        offsets = np.rint(np.linspace(0, max_offset, count)).astype(int).tolist()
        offsets = list(dict.fromkeys(offsets))

    return [
        RollingOrigin(number=number, test_offset=offset, horizon=horizon)
        for number, offset in enumerate(offsets, start=1)
    ]


def expanding_history(
    train: pd.Series,
    test: pd.Series,
    test_offset: int,
    *,
    allow_missing: bool = False,
) -> pd.Series:
    """Return train plus only the test observations revealed before an origin."""
    if test_offset < 0 or test_offset > len(test):
        raise ValueError("test_offset is outside the test split")
    history = pd.concat(
        [train.reset_index(drop=True), test.iloc[:test_offset].reset_index(drop=True)],
        ignore_index=True,
    )
    if not allow_missing and history.isna().any():
        raise ValueError("Rolling-origin history contains missing values")
    return history.astype(float) if not allow_missing else history


def create_lag_features(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised lag features without crossing series boundaries."""
    values = np.asarray(values, dtype=np.float64)
    if lag < 1:
        raise ValueError("lag must be positive")
    if len(values) <= lag:
        return np.empty((0, lag)), np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(values, lag + 1)
    return windows[:, :lag], windows[:, lag]


def create_lag_features_with_missing(
    values: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lag features with missingness indicators for direct-missing XGBoost."""
    values = np.asarray(values, dtype=np.float64)
    if lag < 1:
        raise ValueError("lag must be positive")
    if len(values) <= lag:
        return np.empty((0, lag * 2)), np.empty(0)

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    for idx in range(lag, len(values)):
        window = values[idx - lag : idx]
        mask = np.isfinite(window).astype(float)
        filled = np.where(np.isfinite(window), window, 0.0)
        target = values[idx]
        if not np.isfinite(target):
            continue
        feature_rows.append(np.concatenate([filled, mask]).tolist())
        targets.append(float(target))

    if not feature_rows:
        return np.empty((0, lag * 2)), np.empty(0)
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def fit_predict_xgboost_local(
    history: pd.Series,
    horizon: int,
    params: dict,
    seed: int = 42,
) -> np.ndarray:
    """Refit a local XGBoost model at one origin and forecast recursively."""
    import xgboost as xgb

    lag = int(params.get("lags", 10))
    values = history.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("XGBoost history contains non-finite values")

    features, targets = create_lag_features(values, lag)
    if len(features) == 0:
        raise ValueError(f"Not enough history for XGBoost lag={lag}")

    model = xgb.XGBRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        n_jobs=int(params.get("n_jobs", 1)),
        random_state=int(params.get("random_state", seed)),
        verbosity=0,
    )
    model.fit(features, targets)

    recursive = values.tolist()
    predictions: list[float] = []
    for _ in range(horizon):
        prediction = float(model.predict(np.asarray([recursive[-lag:]], dtype=np.float64))[0])
        predictions.append(prediction)
        recursive.append(prediction)
    return np.asarray(predictions, dtype=np.float64)


def fit_predict_xgboost_direct_missing(
    history: pd.Series,
    horizon: int,
    params: dict,
    seed: int = 42,
) -> np.ndarray:
    """Direct XGBoost on degraded history with native NaNs and missingness flags."""
    import xgboost as xgb

    lag = int(params.get("lags", 10))
    values = history.to_numpy(dtype=np.float64)
    features, targets = create_lag_features_with_missing(values, lag)
    if len(features) == 0:
        raise ValueError(f"Not enough observed targets for direct-missing XGBoost lag={lag}")

    model = xgb.XGBRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        n_jobs=int(params.get("n_jobs", 1)),
        random_state=int(params.get("random_state", seed)),
        verbosity=0,
        missing=np.nan,
    )
    model.fit(features, targets)

    recursive = values.tolist()
    predictions: list[float] = []
    for _ in range(horizon):
        window = np.asarray(recursive[-lag:], dtype=np.float64)
        mask = np.isfinite(window).astype(float)
        filled = np.where(np.isfinite(window), window, 0.0)
        feature = np.concatenate([filled, mask]).reshape(1, -1)
        prediction = float(model.predict(feature)[0])
        predictions.append(prediction)
        recursive.append(prediction)
    return np.asarray(predictions, dtype=np.float64)


def predict_persistence(history: pd.Series, horizon: int) -> np.ndarray:
    """Last observed value repeated for the full horizon."""
    if history.empty:
        raise ValueError("Persistence requires non-empty history")
    last = float(history.iloc[-1])
    if not np.isfinite(last):
        finite = history[np.isfinite(history)]
        if finite.empty:
            raise ValueError("Persistence requires at least one finite history value")
        last = float(finite.iloc[-1])
    return np.full(horizon, last, dtype=np.float64)


def predict_seasonal_naive(
    history: pd.Series,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Seasonal naive forecast using lag ``seasonal_period``."""
    period = max(1, int(seasonal_period))
    values = history.to_numpy(dtype=np.float64)
    if len(values) < period:
        return predict_persistence(history, horizon)

    template = values[-period:]
    if not np.isfinite(template).all():
        filled = template.copy()
        for idx in range(len(filled)):
            if not np.isfinite(filled[idx]):
                source = values[: len(values) - period + idx]
                finite = source[np.isfinite(source)]
                filled[idx] = float(finite[-1]) if len(finite) else 0.0
        template = filled

    return np.asarray([template[(step % period)] for step in range(horizon)], dtype=np.float64)


def cumulative_horizon_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    horizons: Sequence[int],
    train_history: np.ndarray,
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute cumulative metrics for prefixes of one H_max forecast."""
    results: list[dict[str, Any]] = []
    for horizon in sorted({int(h) for h in horizons if int(h) > 0}):
        if horizon > len(actual):
            raise ValueError(
                f"Requested horizon {horizon} exceeds forecast length {len(actual)}"
            )
        actual_h = actual[:horizon]
        predicted_h = predicted[:horizon]
        metrics = compute_prediction_metrics(
            actual_h,
            predicted_h,
            train=train_history,
            metric_keys=list(metric_keys),
        )
        results.append(
            {
                "forecast_horizon": horizon,
                "metric_scope": "cumulative",
                "lead_time_start": 1,
                "lead_time_end": horizon,
                "n_samples": horizon,
                **metrics,
            }
        )
    return results


def lead_time_bin_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    horizons: Sequence[int],
    train_history: np.ndarray,
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute metrics on disjoint lead-time bins derived from sorted horizons."""
    ordered = sorted({int(h) for h in horizons if int(h) > 0})
    results: list[dict[str, Any]] = []
    previous = 0
    for horizon in ordered:
        if horizon > len(actual):
            raise ValueError(
                f"Requested horizon {horizon} exceeds forecast length {len(actual)}"
            )
        start = previous
        end = horizon
        actual_bin = actual[start:end]
        predicted_bin = predicted[start:end]
        if len(actual_bin) == 0:
            continue
        metrics = compute_prediction_metrics(
            actual_bin,
            predicted_bin,
            train=train_history,
            metric_keys=list(metric_keys),
        )
        results.append(
            {
                "forecast_horizon": horizon,
                "metric_scope": "lead_time_bin",
                "lead_time_start": start + 1,
                "lead_time_end": end,
                "n_samples": len(actual_bin),
                **metrics,
            }
        )
        previous = end
    return results


def evaluate_forecast_slices(
    actual: np.ndarray,
    predicted: np.ndarray,
    horizons: Sequence[int],
    train_history: np.ndarray,
    metric_keys: Sequence[str],
    *,
    include_lead_time_bins: bool = True,
) -> list[dict[str, Any]]:
    """Return cumulative and optional lead-time-bin metric rows."""
    rows = cumulative_horizon_metrics(
        actual, predicted, horizons, train_history, metric_keys
    )
    if include_lead_time_bins:
        rows.extend(
            lead_time_bin_metrics(
                actual, predicted, horizons, train_history, metric_keys
            )
        )
    return rows


def resolve_seasonal_period(
    dataset_name: str,
    seasonal_periods: Mapping[str, int],
    default: int = 12,
) -> int:
    stem = dataset_name.replace(".csv", "")
    for key, value in seasonal_periods.items():
        token = str(key).replace(".csv", "")
        if token == stem or token in dataset_name:
            return int(value)
    return int(default)

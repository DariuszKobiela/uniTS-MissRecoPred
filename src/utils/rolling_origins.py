"""Utilities for expanding-history rolling-origin forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    """Spread up to n_origins unique origins across a fixed test split.

    The first origin starts at the train/test boundary. Later origins reveal
    the real test prefix before refitting. If horizon equals test_length, only
    one leakage-free origin exists.
    """
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
) -> pd.Series:
    """Return train plus only the test observations revealed before an origin."""
    if test_offset < 0 or test_offset > len(test):
        raise ValueError("test_offset is outside the test split")
    history = pd.concat(
        [train.reset_index(drop=True), test.iloc[:test_offset].reset_index(drop=True)],
        ignore_index=True,
    )
    if history.isna().any():
        raise ValueError("Rolling-origin history contains missing values")
    return history.astype(float)


def create_lag_features(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised lag features without crossing series boundaries."""
    values = np.asarray(values, dtype=np.float64)
    if lag < 1:
        raise ValueError("lag must be positive")
    if len(values) <= lag:
        return np.empty((0, lag)), np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(values, lag + 1)
    return windows[:, :lag], windows[:, lag]


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

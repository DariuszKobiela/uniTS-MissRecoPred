"""Holt-Winters exponential smoothing for rolling-origin forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _forecast_index(index: pd.Index, horizon: int) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        frequency = index.freq or index.inferred_freq
        if frequency is not None:
            return pd.date_range(index[-1], periods=horizon + 1, freq=frequency)[1:]
    if isinstance(index, pd.RangeIndex):
        return pd.RangeIndex(
            index[-1] + index.step,
            stop=index[-1] + (horizon + 1) * index.step,
            step=index.step,
        )
    if np.issubdtype(index.dtype, np.integer):
        return pd.Index(np.arange(int(index[-1]) + 1, int(index[-1]) + horizon + 1))
    return pd.RangeIndex(len(index), len(index) + horizon)


def _trend_fallback(series: pd.Series, horizon: int) -> np.ndarray:
    recent = series.tail(min(20, len(series)))
    trend = float(np.mean(np.diff(recent))) if len(recent) >= 2 else 0.0
    return float(series.iloc[-1]) + trend * np.arange(1, horizon + 1, dtype=float)


def predict_holt_winters(
    train_series: pd.Series,
    horizon: int,
    seasonal_period: int | None = None,
    trend: str | None = "add",
    seasonal: str | None = "add",
    damped_trend: bool = False,
    max_seasonal_cycles: int | None = 4,
    initialization_method: str = "heuristic",
    random_state: int | None = None,
) -> pd.Series:
    """Fit seasonal Holt-Winters and return an auditable level-scale forecast."""
    del random_state  # Deterministic optimization for fixed input and parameters.
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if seasonal_period is not None and seasonal_period < 2:
        raise ValueError("seasonal_period must be at least 2 or None")
    if max_seasonal_cycles is not None and max_seasonal_cycles < 2:
        raise ValueError("max_seasonal_cycles must be at least 2 or None")

    series = train_series.dropna().astype(float)
    if len(series) < 3:
        raise ValueError("Holt-Winters requires at least three finite observations")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Holt-Winters history contains non-finite observations")

    requested_seasonal = seasonal if seasonal_period is not None else None
    fit_series = series
    if requested_seasonal is not None and max_seasonal_cycles is not None:
        fit_series = series.tail(int(seasonal_period) * int(max_seasonal_cycles))

    mean_value = float(fit_series.mean())
    standard_deviation = float(fit_series.std())
    scale = standard_deviation if np.isfinite(standard_deviation) and standard_deviation > 0 else 1.0
    normalized = (fit_series - mean_value) / scale

    candidates = [
        (requested_seasonal, seasonal_period if requested_seasonal is not None else None),
        (None, None),
    ]
    candidates = list(dict.fromkeys(candidates))
    errors: list[str] = []

    for candidate_number, (candidate_seasonal, candidate_period) in enumerate(candidates):
        try:
            if candidate_period is not None and len(normalized) < 2 * candidate_period:
                raise ValueError(
                    f"seasonal fit requires at least {2 * candidate_period} observations"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ExponentialSmoothing(
                    normalized,
                    trend=trend,
                    damped_trend=bool(damped_trend) if trend is not None else False,
                    seasonal=candidate_seasonal,
                    seasonal_periods=candidate_period,
                    initialization_method=initialization_method,
                ).fit(optimized=True, use_brute=False)
                values = np.asarray(fitted.forecast(horizon), dtype=float)
            if len(values) != horizon or not np.isfinite(values).all():
                raise ValueError("model returned a non-finite or incomplete forecast")
            forecast_values = values * scale + mean_value
            fallback_used = candidate_number > 0
            fallback_type = "nonseasonal_holt" if fallback_used else "none"
            fallback_reason = " | ".join(errors) if fallback_used else ""
            break
        except Exception as exc:
            errors.append(
                f"seasonal={candidate_seasonal}, seasonal_period={candidate_period}: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        forecast_values = _trend_fallback(series, horizon)
        fallback_used = True
        fallback_type = "linear_trend"
        fallback_reason = " | ".join(errors)

    forecast = pd.Series(
        forecast_values,
        index=_forecast_index(train_series.index, horizon),
        name="predicted",
    )
    forecast.attrs.update(
        {
            "fallback_used": fallback_used,
            "fallback_type": fallback_type,
            "fallback_reason": fallback_reason,
        }
    )
    return forecast


train_holt_winters = predict_holt_winters

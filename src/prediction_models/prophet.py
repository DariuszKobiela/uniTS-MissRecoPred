"""Prophet forecasting model used by the rolling-origin experiment."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet


def _silence_backend_logging() -> None:
    """Keep CmdStan worker logs from corrupting the shared progress bar."""
    logger_names = {
        "cmdstanpy",
        "cmdstanpy.model",
        "prophet",
        "prophet.models",
        "prophet.plot",
    }
    logger_names.update(
        name
        for name in logging.Logger.manager.loggerDict
        if name.startswith(("cmdstanpy.", "prophet."))
    )
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False


def _forecast_index(index: pd.Index, horizon: int) -> pd.Index:
    """Extend a regular index when possible; otherwise use positional indices."""
    if isinstance(index, pd.DatetimeIndex):
        frequency = index.freq or index.inferred_freq
        if frequency is not None:
            return pd.date_range(index[-1], periods=horizon + 1, freq=frequency)[1:]
    if isinstance(index, pd.RangeIndex):
        return pd.RangeIndex(index[-1] + index.step, stop=index[-1] + (horizon + 1) * index.step, step=index.step)
    if np.issubdtype(index.dtype, np.integer):
        return pd.Index(np.arange(int(index[-1]) + 1, int(index[-1]) + horizon + 1))
    return pd.RangeIndex(len(index), len(index) + horizon)


def _trend_fallback(series: pd.Series, horizon: int) -> np.ndarray:
    clean = series.dropna().astype(float)
    if len(clean) >= 5:
        recent = clean.tail(min(20, len(clean)))
        trend = float(np.mean(np.diff(recent)))
    elif len(clean) >= 2:
        trend = float(clean.iloc[-1] - clean.iloc[-2])
    else:
        trend = 0.0
    last_value = float(clean.iloc[-1]) if len(clean) else 0.0
    return last_value + trend * np.arange(1, horizon + 1, dtype=float)


def predict_prophet(
    train_series: pd.Series,
    horizon: int,
    yearly_seasonality: bool | str = False,
    weekly_seasonality: bool | str = False,
    daily_seasonality: bool | str = False,
    n_changepoints: int = 5,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
    seasonal_period: int | None = None,
    seasonal_fourier_order: int = 5,
    max_points: int | None = 2000,
    random_state: int | None = None,
) -> pd.Series:
    """Fit Prophet to one-dimensional history and forecast the requested samples."""
    _silence_backend_logging()
    del random_state  # Prophet MAP fitting is deterministic for this configuration.
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if max_points is not None and max_points < 2:
        raise ValueError("max_points must be at least 2 or None")
    if seasonal_period is not None and seasonal_period < 2:
        raise ValueError("seasonal_period must be at least 2 or None")

    series = train_series.dropna().astype(float)
    if len(series) < 2:
        raise ValueError("Prophet requires at least two finite observations")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Prophet history contains non-finite observations")

    positions = np.arange(len(series))
    values = series.to_numpy(dtype=float)
    if max_points is not None and len(series) > max_points:
        selected = np.unique(np.linspace(0, len(series) - 1, max_points, dtype=int))
        positions = positions[selected]
        values = values[selected]

    epoch = pd.Timestamp("2000-01-01")
    history = pd.DataFrame({"ds": epoch + pd.to_timedelta(positions, unit="h"), "y": values})
    future = pd.DataFrame(
        {"ds": epoch + pd.to_timedelta(np.arange(len(series), len(series) + horizon), unit="h")}
    )

    try:
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            n_changepoints=min(int(n_changepoints), max(0, len(history) - 1)),
            changepoint_prior_scale=float(changepoint_prior_scale),
            seasonality_mode=seasonality_mode,
            uncertainty_samples=0,
        )
        if seasonal_period is not None:
            model.add_seasonality(
                name="sample_period",
                period=float(seasonal_period) / 24.0,
                fourier_order=int(seasonal_fourier_order),
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(history)
            forecast_values = model.predict(future)["yhat"].to_numpy(dtype=float)
        if len(forecast_values) != horizon or not np.isfinite(forecast_values).all():
            raise ValueError("model returned a non-finite or incomplete forecast")
        fallback_used = False
        fallback_reason = ""
    except Exception as exc:
        forecast_values = _trend_fallback(series, horizon)
        fallback_used = True
        fallback_reason = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"Prophet fit failed; using trend fallback: {fallback_reason}",
            RuntimeWarning,
            stacklevel=2,
        )

    forecast = pd.Series(forecast_values, index=_forecast_index(train_series.index, horizon), name="predicted")
    forecast.attrs.update(
        {
            "fallback_used": fallback_used,
            "fallback_type": "linear_trend" if fallback_used else "none",
            "fallback_reason": fallback_reason,
        }
    )
    return forecast


train_prophet = predict_prophet

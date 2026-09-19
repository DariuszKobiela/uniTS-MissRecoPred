"""
SARIMAX (Seasonal ARIMA with eXogenous regressors) Prediction Model

SARIMAX is a classical statistical model for time series forecasting that
handles both trend and seasonal components through differencing.

This is a statistical model that does not require GPU.

Requirements:
- statsmodels
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    clean_data = series.dropna().astype(float)
    if len(clean_data) >= 5:
        recent = clean_data.tail(min(20, len(clean_data)))
        weights = np.exp(np.linspace(-1, 0, len(recent) - 1))
        weights /= weights.sum()
        trend = float(np.average(np.diff(recent), weights=weights))
    elif len(clean_data) >= 2:
        trend = float(clean_data.iloc[-1] - clean_data.iloc[-2])
    else:
        trend = 0.0
    last_value = float(clean_data.iloc[-1]) if len(clean_data) else 0.0
    return np.asarray([last_value + trend * (step + 1) for step in range(horizon)])


def predict_sarimax(train_series: pd.Series, horizon: int,
                    order: tuple = (1, 1, 1),
                    seasonal_order: tuple = (0, 0, 0, 0),  # Disabled by default for speed
                    random_state: int = None) -> pd.Series:
    """
    Train SARIMAX and forecast in the original data scale.

    The returned Series includes audit metadata in ``attrs``. In particular,
    ``fallback_used`` and ``fallback_reason`` make degraded model fits visible
    to experiment reporting.
    """
    del random_state  # SARIMAX fitting is deterministic for fixed inputs.
    if horizon < 1:
        raise ValueError("horizon must be positive")
    order = tuple(order)
    seasonal_order = tuple(seasonal_order)
    if len(order) != 3 or len(seasonal_order) != 4:
        raise ValueError("order and seasonal_order must have lengths 3 and 4")

    series = train_series.dropna().astype(float).copy()
    if len(series) < 3:
        raise ValueError("SARIMAX requires at least three finite observations")
    mean_val = float(series.mean())
    std_val = float(series.std())
    scale = std_val if np.isfinite(std_val) and std_val > 0 else 1.0
    normalized = (series - mean_val) / scale

    requested = {"order": order, "seasonal_order": seasonal_order}
    candidates = [
        requested,
        {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0)},
        {"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0)},
        {"order": (0, 1, 1), "seasonal_order": (0, 0, 0, 0)},
    ]
    # Do not retry an identical configuration.
    candidates = list(dict.fromkeys((c["order"], c["seasonal_order"]) for c in candidates))
    errors = []

    for candidate_number, (candidate_order, candidate_seasonal) in enumerate(candidates):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = SARIMAX(
                    normalized,
                    order=candidate_order,
                    seasonal_order=candidate_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    simple_differencing=False,
                ).fit(disp=False, maxiter=100, method="powell")
            values = np.asarray(fitted.forecast(steps=horizon), dtype=float)
            if len(values) != horizon or not np.isfinite(values).all():
                raise ValueError("model returned a non-finite or incomplete forecast")
            forecast = pd.Series(
                values * scale + mean_val,
                index=_forecast_index(train_series.index, horizon),
                name="predicted",
            )
            fallback_used = candidate_number > 0
            forecast.attrs.update(
                {
                    "fallback_used": fallback_used,
                    "fallback_type": "simplified_sarimax" if fallback_used else "none",
                    "fallback_reason": " | ".join(errors) if fallback_used else "",
                    "requested_order": str(order),
                    "requested_seasonal_order": str(seasonal_order),
                    "fitted_order": str(candidate_order),
                    "fitted_seasonal_order": str(candidate_seasonal),
                }
            )
            if fallback_used:
                warnings.warn(
                    f"SARIMAX used a simplified fallback: {forecast.attrs}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return forecast
        except Exception as exc:
            errors.append(
                f"order={candidate_order}, seasonal_order={candidate_seasonal}: "
                f"{type(exc).__name__}: {exc}"
            )

    forecast = pd.Series(
        _trend_fallback(series, horizon),
        index=_forecast_index(train_series.index, horizon),
        name="predicted",
    )
    forecast.attrs.update(
        {
            "fallback_used": True,
            "fallback_type": "linear_trend",
            "fallback_reason": " | ".join(errors),
            "requested_order": str(order),
            "requested_seasonal_order": str(seasonal_order),
            "fitted_order": "",
            "fitted_seasonal_order": "",
        }
    )
    warnings.warn(
        f"All SARIMAX configurations failed; using trend fallback: {forecast.attrs['fallback_reason']}",
        RuntimeWarning,
        stacklevel=2,
    )
    return forecast


# Alias for backward compatibility
train_sarimax = predict_sarimax

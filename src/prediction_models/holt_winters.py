"""
Holt-Winters Exponential Smoothing Prediction Model

Holt-Winters is a classical statistical method for time series forecasting
that captures level, trend, and seasonal components.

This is a traditional statistical model that does not require GPU or
deep learning frameworks. It fits quickly on the provided data.

Requirements:
- statsmodels
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing


def predict_holt_winters(train_series: pd.Series, horizon: int,
                         seasonal_periods: int = 168,
                         trend: str = "add",
                         seasonal: str = "add",
                         random_state: int = None) -> pd.Series:
    """
    Trains a Holt-Winters Exponential Smoothing model and predicts future values.
    
    Holt-Winters method decomposes time series into three components:
    - Level: the average value in the series
    - Trend: the increasing or decreasing value in the series
    - Seasonality: the repeating short-term cycle in the series
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    seasonal_periods : int
        Number of periods in a complete seasonal cycle
        Default: 168 (weekly seasonality for hourly data: 7 days * 24 hours)
    trend : str
        Type of trend component: "add", "mul", or None
    seasonal : str
        Type of seasonal component: "add", "mul", or None
    random_state : int
        Ignored (model is deterministic)
        
    Returns
    -------
    pd.Series
        Predicted values with appropriate index
        
    Notes
    -----
    - No training required (statistical estimation)
    - Very fast compared to deep learning models
    - Works well for data with clear trend and seasonal patterns
    - Deterministic output (no randomness)
    """
    
    try:
        # 1. Create a datetime index (statsmodels requirement for seasonal)
        series = train_series.copy()
        series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='H')
        
        # 2. Adjust seasonal_periods if series is too short
        if len(series) < 2 * seasonal_periods:
            # Not enough data for seasonal pattern detection
            # Try with smaller seasonal period or no seasonality
            if len(series) >= 48:  # At least 2 days of hourly data
                seasonal_periods = 24  # Daily seasonality
            else:
                seasonal = None  # Disable seasonality
        
        # 3. Fit the model
        if seasonal is not None:
            model = ExponentialSmoothing(
                series,
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal,
                initialization_method="estimated",
            ).fit(optimized=True)
        else:
            # Simple exponential smoothing without seasonality
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        
        # 4. Generate forecast
        forecast = model.forecast(steps=horizon)
        
        # 5. Check for valid forecast
        if not np.isfinite(forecast.values).all():
            raise ValueError("Forecast contains inf or nan values")
        
        # 6. Convert index back to integer
        last_idx = train_series.index[-1]
        forecast.index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return forecast
        
    except Exception as e:
        print(f"Warning: Holt-Winters prediction failed: {e}")
        # Fallback: simple exponential smoothing or trend extrapolation
        try:
            series = train_series.copy()
            series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='H')
            
            # Try simple model without seasonality
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            
            forecast = model.forecast(steps=horizon)
            last_idx = train_series.index[-1]
            forecast.index = range(last_idx + 1, last_idx + 1 + horizon)
            return forecast
            
        except Exception:
            # Final fallback: trend extrapolation
            clean_data = train_series.dropna()
            if len(clean_data) >= 2:
                trend_val = np.mean(np.diff(clean_data.tail(min(10, len(clean_data)))))
                last_value = clean_data.iloc[-1]
                forecast_values = [last_value + trend_val * (i + 1) for i in range(horizon)]
            else:
                last_value = clean_data.iloc[-1] if len(clean_data) > 0 else 0
                forecast_values = [last_value] * horizon
            
            last_idx = train_series.index[-1]
            forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
            return pd.Series(forecast_values, index=forecast_index, name='predicted')


# Alias for backward compatibility
train_holt_winters = predict_holt_winters

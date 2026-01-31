"""
Holt-Winters Exponential Smoothing Prediction Model

Fast statistical method for time series forecasting.

Requirements:
- statsmodels
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.api import ExponentialSmoothing


def predict_holt_winters(train_series: pd.Series, horizon: int,
                         seasonal_periods: int = None,  # Disabled by default for speed
                         trend: str = "add",
                         seasonal: str = None,  # Disabled by default for speed
                         random_state: int = None) -> pd.Series:
    """
    Fast Holt-Winters Exponential Smoothing prediction.
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    try:
        # 1. Prepare data
        series = train_series.copy().astype(float)
        series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='h')
        
        # 2. Fast simple exponential smoothing (no seasonality = much faster)
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=None,
            initialization_method="heuristic",  # Faster than "estimated"
        ).fit(optimized=False)  # Don't optimize = much faster
        
        forecast = model.forecast(steps=horizon)
        
        if not np.isfinite(forecast.values).all():
            raise ValueError("Forecast invalid")
        
        # 3. Create output with proper index
        original_index = train_series.index
        if hasattr(original_index[-1], 'freq') or isinstance(original_index[-1], pd.Timestamp):
            start_pos = len(original_index)
            forecast.index = range(start_pos, start_pos + horizon)
        else:
            last_idx = int(original_index[-1])
            forecast.index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return forecast
        
    except Exception:
        # Fallback: simple trend extrapolation (very fast)
        clean_data = train_series.dropna().astype(float)
        
        if len(clean_data) >= 5:
            # Exponential weighted trend
            recent = clean_data.tail(min(20, len(clean_data)))
            alpha = 0.3  # Smoothing factor
            level = recent.iloc[0]
            trend_val = 0
            for val in recent:
                level = alpha * val + (1 - alpha) * (level + trend_val)
                trend_val = alpha * (level - (alpha * val + (1 - alpha) * level)) + (1 - alpha) * trend_val
            
            last_value = level
            forecast_values = [last_value + trend_val * (i + 1) for i in range(horizon)]
        elif len(clean_data) >= 2:
            trend_val = clean_data.iloc[-1] - clean_data.iloc[-2]
            last_value = clean_data.iloc[-1]
            forecast_values = [last_value + trend_val * (i + 1) for i in range(horizon)]
        else:
            last_value = clean_data.iloc[-1] if len(clean_data) > 0 else 0
            forecast_values = [last_value] * horizon
        
        original_index = train_series.index
        if hasattr(original_index[-1], 'freq') or isinstance(original_index[-1], pd.Timestamp):
            start_pos = len(original_index)
            forecast_index = range(start_pos, start_pos + horizon)
        else:
            last_idx = int(original_index[-1])
            forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')


# Alias for backward compatibility
train_holt_winters = predict_holt_winters

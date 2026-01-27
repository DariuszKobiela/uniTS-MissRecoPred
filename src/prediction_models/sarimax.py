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
from statsmodels.tsa.statespace.sarimax import SARIMAX


def predict_sarimax(train_series: pd.Series, horizon: int,
                    order: tuple = (1, 1, 1),
                    seasonal_order: tuple = (1, 1, 1, 12),
                    random_state: int = None) -> pd.Series:
    """
    Trains a SARIMAX model and predicts future values.
    
    SARIMAX combines:
    - AR (AutoRegressive): dependency on past values
    - I (Integrated): differencing to achieve stationarity
    - MA (Moving Average): dependency on past forecast errors
    - Seasonal components for periodic patterns
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    order : tuple
        ARIMA order (p, d, q): AR order, differencing, MA order
    seasonal_order : tuple
        Seasonal order (P, D, Q, s): seasonal AR, differencing, MA, period
    random_state : int
        Ignored (SARIMAX is deterministic)
        
    Returns
    -------
    pd.Series
        Predicted values with appropriate index
        
    Notes
    -----
    - No pre-trained model required
    - Fast fitting for most time series
    - Deterministic output
    - Handles extreme values by clipping
    """
    
    try:
        # 1. Create datetime index (statsmodels requirement)
        series = train_series.copy()
        series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='H')
        
        # 2. Handle extreme values that can cause convergence issues
        mean_val = series.mean()
        std_val = series.std()
        
        if std_val > 0:
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            series = series.clip(lower=lower_bound, upper=upper_bound)
        
        # 3. Try full SARIMAX model
        try:
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                suppress_warnings=True
            ).fit(disp=False)
            
            forecast = model.forecast(steps=horizon)
            
            if not np.isfinite(forecast.values).all():
                raise ValueError("Forecast contains inf or nan values")
                
        except Exception:
            # 4. Fallback: simpler ARIMA without seasonal component
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=True,
                enforce_invertibility=True,
                suppress_warnings=True
            ).fit(disp=False, maxiter=50)
            
            forecast = model.forecast(steps=horizon)
            
            if not np.isfinite(forecast.values).all():
                raise ValueError("Forecast contains inf or nan values")
        
        # 5. Convert index to integer
        last_idx = train_series.index[-1]
        forecast.index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return forecast
        
    except Exception as e:
        print(f"Warning: SARIMAX prediction failed: {e}")
        # Fallback: simple trend extrapolation
        clean_data = train_series.dropna()
        if len(clean_data) >= 2:
            trend = np.mean(np.diff(clean_data.tail(min(10, len(clean_data)))))
            last_value = clean_data.iloc[-1]
            forecast_values = [last_value + trend * (i + 1) for i in range(horizon)]
        else:
            last_value = clean_data.iloc[-1] if len(clean_data) > 0 else 0
            forecast_values = [last_value] * horizon
        
        last_idx = train_series.index[-1]
        forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        return pd.Series(forecast_values, index=forecast_index, name='predicted')


# Alias for backward compatibility
train_sarimax = predict_sarimax

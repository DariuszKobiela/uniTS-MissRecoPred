"""
Prophet Prediction Model

Prophet is a forecasting procedure developed by Facebook that is robust to
missing data and shifts in the trend, and typically handles outliers well.

This is a statistical model that does not require GPU. It automatically
detects seasonality patterns.

Requirements:
- prophet
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# Suppress Prophet's verbose output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


def predict_prophet(train_series: pd.Series, horizon: int,
                    yearly_seasonality: bool = False,
                    weekly_seasonality: bool = False,
                    daily_seasonality: bool = False,
                    random_state: int = None) -> pd.Series:
    """
    Trains a Prophet model and predicts future values.
    
    Prophet is designed for forecasting with:
    - Automatic detection of changepoints
    - Flexible seasonality modeling
    - Robust handling of missing data and outliers
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    yearly_seasonality : bool
        Whether to detect yearly seasonality
    weekly_seasonality : bool
        Whether to detect weekly seasonality
    daily_seasonality : bool
        Whether to detect daily seasonality
    random_state : int
        Ignored (Prophet's point forecast is deterministic)
        
    Returns
    -------
    pd.Series
        Predicted values with appropriate index
        
    Notes
    -----
    - No pre-trained model required
    - Fast fitting compared to deep learning models
    - Deterministic output (no randomness in point forecasts)
    - Automatically handles missing values and outliers
    """
    
    try:
        # 1. Convert to Prophet format: DataFrame with 'ds' and 'y' columns
        df = pd.DataFrame({'y': train_series.values})
        
        # Create datetime column (Prophet requirement)
        df['ds'] = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
        
        # 2. Initialize and train the model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        model.fit(df)
        
        # 3. Create future dataframe and predict
        future = model.make_future_dataframe(periods=horizon, freq='H')
        forecast = model.predict(future)
        
        # 4. Extract forecast values
        forecast_values = forecast['yhat'].iloc[-horizon:].values
        
        # Check for valid forecast
        if not np.isfinite(forecast_values).all():
            raise ValueError("Forecast contains inf or nan values")
        
        # 5. Convert to Series with integer index
        last_idx = train_series.index[-1]
        forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: Prophet prediction failed: {e}")
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
train_prophet = predict_prophet

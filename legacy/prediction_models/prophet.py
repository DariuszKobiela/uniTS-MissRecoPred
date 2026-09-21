"""
Prophet Prediction Model (Optimized for Speed)

Requirements:
- prophet
"""

import pandas as pd
import numpy as np
import logging
import warnings
import os

# Suppress ALL Prophet/cmdstanpy output before importing
os.environ['CMDSTAN_VERBOSE'] = 'false'
os.environ['STAN_NUM_THREADS'] = '1'
logging.getLogger('prophet').setLevel(logging.CRITICAL)
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
logging.getLogger('cmdstan').setLevel(logging.CRITICAL)

from prophet import Prophet


def predict_prophet(train_series: pd.Series, horizon: int,
                    yearly_seasonality: bool = False,
                    weekly_seasonality: bool = False,
                    daily_seasonality: bool = False,
                    random_state: int = None) -> pd.Series:
    """Ultra-fast Prophet prediction with minimal settings."""
    # Suppress ALL warnings and logging
    warnings.filterwarnings('ignore')
    logging.disable(logging.CRITICAL)
    
    try:
        # 1. Subsample data if too long (Prophet is slow with large datasets)
        series = train_series.copy().astype(float)
        max_points = 500  # Limit data points for speed
        if len(series) > max_points:
            # Take evenly spaced samples
            indices = np.linspace(0, len(series) - 1, max_points, dtype=int)
            series = series.iloc[indices]
        
        # 2. Convert to Prophet format
        df = pd.DataFrame({
            'y': series.values,
            'ds': pd.date_range(start='2000-01-01', periods=len(series), freq='h')
        })
        
        # 3. Initialize model with MINIMAL settings for speed
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            n_changepoints=5,  # Reduced from default 25
            changepoint_prior_scale=0.5,
            uncertainty_samples=0,  # No uncertainty = faster
        )
        
        # Fit with suppressed output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        # 4. Create minimal future dataframe
        last_date = df['ds'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                     periods=horizon, freq='h')
        future = pd.DataFrame({'ds': future_dates})
        
        # 5. Predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(future)
        
        forecast_values = forecast['yhat'].values
        
        if not np.isfinite(forecast_values).all():
            raise ValueError("Forecast invalid")
        
        # 6. Create output index
        original_index = train_series.index
        if hasattr(original_index[-1], 'freq') or isinstance(original_index[-1], pd.Timestamp):
            start_pos = len(original_index)
            forecast_index = range(start_pos, start_pos + horizon)
        else:
            last_idx = int(original_index[-1])
            forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception:
        # Fallback: exponential smoothing (very fast)
        clean_data = train_series.dropna().astype(float)
        
        if len(clean_data) >= 5:
            alpha = 0.3
            level = clean_data.iloc[-1]
            trend = (clean_data.iloc[-1] - clean_data.iloc[-min(10, len(clean_data))]) / min(10, len(clean_data))
            forecast_values = [level + trend * (i + 1) for i in range(horizon)]
        elif len(clean_data) >= 2:
            trend = clean_data.iloc[-1] - clean_data.iloc[-2]
            forecast_values = [clean_data.iloc[-1] + trend * (i + 1) for i in range(horizon)]
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
    
    finally:
        # Re-enable logging
        logging.disable(logging.NOTSET)


# Alias for backward compatibility
train_prophet = predict_prophet

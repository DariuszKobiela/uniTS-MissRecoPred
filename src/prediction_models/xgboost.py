"""
XGBoost Prediction Model

XGBoost is a gradient boosting algorithm that uses decision trees as base learners.
For time series, we create lag features and use recursive forecasting.

This is a machine learning model that trains quickly and doesn't require GPU.

Requirements:
- xgboost
"""

import pandas as pd
import numpy as np
import xgboost as xgb


def create_lag_features(series: pd.Series, lag: int) -> pd.DataFrame:
    """
    Create lag features for time series prediction.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    lag : int
        Number of lag features to create
        
    Returns
    -------
    pd.DataFrame
        DataFrame with target 'y' and lag features
    """
    df = pd.DataFrame(series.values, columns=['y'])
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df.dropna(inplace=True)
    return df


def predict_xgboost(train_series: pd.Series, horizon: int,
                    lag: int = 10,
                    n_estimators: int = 100,
                    max_depth: int = 6,
                    learning_rate: float = 0.1,
                    random_state: int = None) -> pd.Series:
    """
    Trains an XGBoost model and predicts future values using recursive forecasting.
    
    This model uses:
    - Lag features as input (previous values)
    - Recursive forecasting: each prediction becomes input for next step
    - Gradient boosting with decision trees
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    lag : int
        Number of past values to use as features
    n_estimators : int
        Number of boosting rounds (trees)
    max_depth : int
        Maximum depth of each tree
    learning_rate : float
        Learning rate (shrinkage)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Predicted values with appropriate index
        
    Notes
    -----
    - Model is trained from scratch (no pre-trained weights required)
    - Fast training compared to deep learning
    - Recursive forecasting: predicts one step at a time, using previous
      predictions as input for next step
    - Good baseline model for time series
    """
    
    try:
        # 1. Create lag features
        df = create_lag_features(train_series, lag)
        X, y = df.drop('y', axis=1), df['y']
        
        # 2. Train the model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0
        )
        model.fit(X, y)
        
        # 3. Recursive forecasting (step by step)
        history = list(train_series.values)
        predictions = []
        
        for _ in range(horizon):
            # Create input features from most recent 'lag' values
            input_features = np.array(history[-lag:]).reshape(1, -1)
            
            # Predict next step
            pred = model.predict(input_features)[0]
            predictions.append(pred)
            
            # Add prediction to history for next step
            history.append(pred)
        
        # 4. Return forecast with integer index
        last_idx = train_series.index[-1]
        forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return pd.Series(predictions, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: XGBoost prediction failed: {e}")
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
train_xgboost = predict_xgboost

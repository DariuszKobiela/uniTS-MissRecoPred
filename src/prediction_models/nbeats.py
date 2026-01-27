"""
N-BEATS (Neural Basis Expansion Analysis for Time Series) Prediction Model

N-BEATS is a deep learning architecture specifically designed for time series
forecasting, featuring stacks of basic blocks with skip connections.

This implementation uses the Darts library with PyTorch backend.
The model is trained from scratch on the provided training data.

Requirements:
- darts
- pytorch-lightning
- torch
"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
NBEATS_INPUT_LEN = 24
NBEATS_OUTPUT_LEN = 12
NBEATS_NUM_STACKS = 30
NBEATS_NUM_BLOCKS = 1
NBEATS_NUM_LAYERS = 4
NBEATS_LAYER_WIDTHS = 256


def predict_nbeats(train_series: pd.Series, horizon: int,
                   input_chunk_length: int = NBEATS_INPUT_LEN,
                   output_chunk_length: int = None,
                   generic_architecture: bool = True,
                   num_stacks: int = NBEATS_NUM_STACKS,
                   num_blocks: int = NBEATS_NUM_BLOCKS,
                   num_layers: int = NBEATS_NUM_LAYERS,
                   layer_widths: int = NBEATS_LAYER_WIDTHS,
                   epochs: int = 100,
                   random_state: int = None) -> pd.Series:
    """
    Trains an N-BEATS model and predicts future values.
    
    N-BEATS architecture features:
    - Stacks of basic blocks with skip connections
    - Doubly residual stacking principle
    - Optional interpretable architecture with trend/seasonality decomposition
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    input_chunk_length : int
        Number of past time steps to use as input (lookback window)
    output_chunk_length : int
        Number of steps to forecast in one pass. If None, uses min(horizon, 12)
    generic_architecture : bool
        If True, uses generic architecture. If False, uses interpretable
    num_stacks : int
        Number of stacks in the model
    num_blocks : int
        Number of blocks per stack
    num_layers : int
        Number of fully connected layers in each block
    layer_widths : int
        Width of fully connected layers
    epochs : int
        Maximum number of training epochs
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Predicted values with appropriate index
        
    Notes
    -----
    - Model is trained from scratch (no pre-trained weights required)
    - Uses early stopping to prevent overfitting
    - Can use interpretable architecture for better explainability
    """
    
    try:
        # Set output_chunk_length
        if output_chunk_length is None:
            output_chunk_length = min(horizon, NBEATS_OUTPUT_LEN)
        
        # 1. Create TimeSeries with datetime index
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
        full_ts = TimeSeries.from_times_and_values(
            times=date_index, 
            values=train_series.values, 
            freq='H'
        )
        
        # 2. Split into training and validation sets
        train_split_point = int(len(full_ts) * 0.8)
        ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]
        
        # Adjust input_chunk_length if series is too short
        if len(ts) < input_chunk_length:
            input_chunk_length = max(10, len(ts) // 2)
        
        # 3. Define EarlyStopping callback
        early_stopper = EarlyStopping(
            "val_loss", 
            patience=10, 
            min_delta=0.001, 
            verbose=False
        )
        
        # 4. Initialize N-BEATS model
        model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=generic_architecture,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            expansion_coefficient_dim=5,
            trend_polynomial_degree=2,
            dropout=0.1,
            activation="ReLU",
            batch_size=128,
            n_epochs=epochs,
            random_state=random_state,
            pl_trainer_kwargs={
                "callbacks": [early_stopper],
                "accelerator": "auto",
                "enable_progress_bar": False,
                "enable_model_summary": False
            },
            force_reset=True,
            save_checkpoints=False
        )
        
        # 5. Train the model
        model.fit(ts, val_series=val_ts, verbose=False)
        
        # 6. Generate forecast
        prediction = model.predict(n=horizon)
        
        # 7. Convert back to pd.Series with integer index
        forecast_values = prediction.values().flatten()
        last_idx = train_series.index[-1]
        forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: N-BEATS prediction failed: {e}")
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


def predict_nbeats_interpretable(train_series: pd.Series, horizon: int,
                                  random_state: int = None) -> pd.Series:
    """
    Trains an interpretable N-BEATS model with separate trend and seasonality stacks.
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data
    horizon : int
        Number of steps to forecast
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Predicted values with appropriate index
    """
    return predict_nbeats(
        train_series, 
        horizon, 
        generic_architecture=False,
        num_stacks=2,
        num_blocks=3,
        random_state=random_state
    )


# Aliases for backward compatibility
train_nbeats = predict_nbeats
train_nbeats_interpretable = predict_nbeats_interpretable

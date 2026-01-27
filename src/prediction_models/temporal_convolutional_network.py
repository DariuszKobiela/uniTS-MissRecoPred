"""
TCN (Temporal Convolutional Network) Prediction Model

TCN uses dilated causal convolutions to capture long-range temporal dependencies
while maintaining computational efficiency.

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
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
TCN_INPUT_LEN = 100
TCN_OUTPUT_LEN = 10


def predict_tcn(train_series: pd.Series, horizon: int,
                input_chunk_length: int = TCN_INPUT_LEN,
                output_chunk_length: int = TCN_OUTPUT_LEN,
                epochs: int = 100,
                random_state: int = None) -> pd.Series:
    """
    Trains a Temporal Convolutional Network (TCN) model and predicts future values.
    
    TCN architecture features:
    - Dilated causal convolutions for capturing long-range dependencies
    - Residual connections for training deeper networks
    - Parallel processing (unlike sequential RNNs)
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    input_chunk_length : int
        Number of past time steps to use as input
    output_chunk_length : int
        Number of steps to predict in one forward pass
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
    - Faster than RNNs due to parallel processing
    """
    
    try:
        # 1. Create TimeSeries with datetime index (Darts requirement)
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
        full_ts = TimeSeries.from_times_and_values(
            times=date_index, 
            values=train_series.values, 
            freq='H'
        )
        
        # 2. Split into training and validation sets (80/20)
        train_split_point = int(len(full_ts) * 0.8)
        ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]
        
        # Adjust input_chunk_length if series is too short
        if len(ts) < input_chunk_length:
            input_chunk_length = max(10, len(ts) // 2)
        
        # Adjust output_chunk_length
        output_chunk_length = min(output_chunk_length, horizon, len(ts) // 4)
        
        # 3. Define EarlyStopping callback
        early_stopper = EarlyStopping(
            "val_loss", 
            patience=5, 
            min_delta=0.005, 
            verbose=False
        )
        
        # 4. Initialize TCN model
        model = TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            kernel_size=3,
            num_filters=64,
            dilation_base=2,
            dropout=0.1,
            batch_size=32,
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
        print(f"Warning: TCN prediction failed: {e}")
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
train_tcn = predict_tcn

"""
Transformer (Temporal Fusion Transformer) Prediction Model

TFT is a state-of-the-art attention-based architecture for time series forecasting.

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
from darts.models import TFTModel
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
TFT_INPUT_LEN = 24
TFT_OUTPUT_LEN = 12
TFT_HIDDEN_SIZE = 64
TFT_LSTM_LAYERS = 1
TFT_NUM_ATTENTION_HEADS = 4


def predict_transformer(train_series: pd.Series, horizon: int,
                        input_chunk_length: int = TFT_INPUT_LEN,
                        output_chunk_length: int = None,
                        hidden_size: int = TFT_HIDDEN_SIZE,
                        lstm_layers: int = TFT_LSTM_LAYERS,
                        num_attention_heads: int = TFT_NUM_ATTENTION_HEADS,
                        epochs: int = 100,
                        random_state: int = None) -> pd.Series:
    """
    Trains a Temporal Fusion Transformer (TFT) model and predicts future values.
    
    TFT architecture features:
    - Multi-head self-attention for capturing long-range dependencies
    - Variable selection networks for interpretability
    - Gated residual networks for information flow
    - Quantile regression for probabilistic forecasts
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    input_chunk_length : int
        Number of past time steps to use as input (encoder length)
    output_chunk_length : int
        Number of steps to forecast in one pass. If None, uses min(horizon, 12)
    hidden_size : int
        Hidden state size of the model
    lstm_layers : int
        Number of LSTM layers in encoder/decoder
    num_attention_heads : int
        Number of attention heads
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
    - State-of-the-art performance on many time series benchmarks
    """
    
    try:
        # Set output_chunk_length
        if output_chunk_length is None:
            output_chunk_length = min(horizon, TFT_OUTPUT_LEN)
        
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
        
        # 4. Initialize TFT model
        model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=0.1,
            batch_size=32,
            n_epochs=epochs,
            add_relative_index=True,
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
        print(f"Warning: Transformer prediction failed: {e}")
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


# Aliases
train_tft = predict_transformer
train_transformer = predict_transformer
predict_tft = predict_transformer

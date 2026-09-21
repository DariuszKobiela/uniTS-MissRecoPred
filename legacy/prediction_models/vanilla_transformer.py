"""
Vanilla Transformer Prediction Model

A general-purpose Transformer architecture adapted for time series forecasting.
Uses encoder-decoder self-attention without the specialized components of TFT.

Simpler and faster than TFT, but may not capture complex temporal patterns as well.

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
from darts.models import TransformerModel
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
TRANSFORMER_INPUT_LEN = 24
TRANSFORMER_OUTPUT_LEN = 12
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_ENCODER_LAYERS = 2
TRANSFORMER_NUM_DECODER_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128


def predict_transformer(train_series: pd.Series, horizon: int,
                        input_chunk_length: int = TRANSFORMER_INPUT_LEN,
                        output_chunk_length: int = None,
                        d_model: int = TRANSFORMER_D_MODEL,
                        nhead: int = TRANSFORMER_NHEAD,
                        num_encoder_layers: int = TRANSFORMER_NUM_ENCODER_LAYERS,
                        num_decoder_layers: int = TRANSFORMER_NUM_DECODER_LAYERS,
                        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
                        epochs: int = 100,
                        random_state: int = None) -> pd.Series:
    """
    Trains a vanilla Transformer model and predicts future values.
    
    Transformer architecture features:
    - Multi-head self-attention in encoder and decoder
    - Positional encoding for temporal information
    - Simpler than TFT, faster training
    
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
    d_model : int
        The number of expected features in the encoder/decoder inputs
    nhead : int
        Number of attention heads
    num_encoder_layers : int
        Number of encoder layers
    num_decoder_layers : int
        Number of decoder layers
    dim_feedforward : int
        Dimension of the feedforward network
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
    - Faster and lighter than TFT
    """
    
    try:
        # Set output_chunk_length
        if output_chunk_length is None:
            output_chunk_length = min(horizon, TRANSFORMER_OUTPUT_LEN)
        
        # 1. Create TimeSeries with datetime index
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='h')
        full_ts = TimeSeries.from_times_and_values(
            times=date_index, 
            values=train_series.values, 
            freq='h'
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
        
        # 4. Initialize Transformer model
        model = TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
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
train_transformer = predict_transformer

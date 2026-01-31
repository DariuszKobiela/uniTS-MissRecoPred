"""
LSTM (Long Short-Term Memory) Prediction Model

LSTM is a type of recurrent neural network architecture designed to capture
long-term dependencies in sequential data through its gating mechanisms.

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
from darts.models import RNNModel
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
LSTM_INPUT_LEN = 100
LSTM_HIDDEN_DIM = 32
LSTM_LAYERS = 2


def predict_lstm(train_series: pd.Series, horizon: int,
                 input_chunk_length: int = LSTM_INPUT_LEN,
                 hidden_dim: int = LSTM_HIDDEN_DIM,
                 n_rnn_layers: int = LSTM_LAYERS,
                 epochs: int = 100,
                 random_state: int = None) -> pd.Series:
    """
    Trains an LSTM model on training data and predicts future values.
    
    LSTM (Long Short-Term Memory) networks are capable of learning long-term
    dependencies through their cell state and gating mechanisms:
    - Forget gate: decides what information to discard
    - Input gate: decides what new information to store
    - Output gate: decides what to output based on cell state
    
    The model is trained from scratch on the provided data. For each prediction,
    the model uses autoregressive approach internally (predicting step by step).
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    input_chunk_length : int
        Number of past time steps to use as input context
    hidden_dim : int
        Size of LSTM hidden state
    n_rnn_layers : int
        Number of stacked LSTM layers
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
    - Autoregressive prediction: each step uses previous predictions as input
    """
    
    try:
        # 1. Create TimeSeries with datetime index (Darts requirement)
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='h')
        full_ts = TimeSeries.from_times_and_values(
            times=date_index, 
            values=train_series.values, 
            freq='h'
        )
        
        # 2. Split into training and validation sets (80/20)
        train_split_point = int(len(full_ts) * 0.8)
        ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]
        
        # Adjust input_chunk_length if series is too short
        if len(ts) < input_chunk_length:
            input_chunk_length = max(10, len(ts) // 2)
        
        # 3. Define EarlyStopping callback
        early_stopper = EarlyStopping(
            "val_loss", 
            patience=5, 
            min_delta=0.005, 
            verbose=False
        )
        
        # 4. Initialize LSTM model
        model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            training_length=min(24, input_chunk_length),
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
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
        
        # 6. Generate forecast (autoregressive prediction internally)
        prediction = model.predict(n=horizon)
        
        # 7. Convert back to pd.Series with integer index
        forecast_values = prediction.values().flatten()
        last_idx = train_series.index[-1]
        forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: LSTM prediction failed: {e}")
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

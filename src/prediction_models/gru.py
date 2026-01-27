"""
GRU (Gated Recurrent Unit) Prediction Model

GRU is a type of recurrent neural network architecture that simplifies LSTM
by combining the forget and input gates into a single update gate.

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
GRU_INPUT_LEN = 100
GRU_HIDDEN_DIM = 32
GRU_LAYERS = 2


def predict_gru(train_series: pd.Series, horizon: int,
                input_chunk_length: int = GRU_INPUT_LEN,
                hidden_dim: int = GRU_HIDDEN_DIM,
                n_rnn_layers: int = GRU_LAYERS,
                epochs: int = 100,
                random_state: int = None) -> pd.Series:
    """
    Trains a GRU model on training data and predicts future values.
    
    GRU (Gated Recurrent Unit) is a simplified variant of LSTM with:
    - Update gate: combines forget and input gates
    - Reset gate: controls how much past information to forget
    
    GRU has fewer parameters than LSTM, making it faster to train while
    often achieving comparable performance.
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data (complete, without missing values)
    horizon : int
        Number of future steps to predict
    input_chunk_length : int
        Number of past time steps to use as input context
    hidden_dim : int
        Size of GRU hidden state
    n_rnn_layers : int
        Number of stacked GRU layers
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
    - Faster training than LSTM due to simpler architecture
    - Autoregressive prediction: each step uses previous predictions as input
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
        
        # 3. Define EarlyStopping callback
        early_stopper = EarlyStopping(
            "val_loss", 
            patience=5, 
            min_delta=0.005, 
            verbose=False
        )
        
        # 4. Initialize GRU model
        model = RNNModel(
            model="GRU",
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
        print(f"Warning: GRU prediction failed: {e}")
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

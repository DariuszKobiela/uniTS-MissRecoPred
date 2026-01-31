"""
Prediction Models Module

This module provides various time series prediction models for forecasting
future values based on historical training data.

All models follow a consistent interface:
    predict_XXX(train_series: pd.Series, horizon: int, random_state: int = None) -> pd.Series

Where:
    - train_series: Historical data to train on (complete, without NaN)
    - horizon: Number of future steps to predict
    - random_state: Random seed for reproducibility (ignored by deterministic models)
    - Returns: pd.Series with predicted values

Model Categories:
-----------------
1. Statistical Models (fast, no GPU required):
   - holt_winters: Exponential smoothing with trend and seasonality
   - prophet: Facebook's forecasting procedure
   - sarimax: Seasonal ARIMA model

2. Machine Learning Models (fast, no GPU required):
   - xgboost: Gradient boosting with lag features

3. Deep Learning Models (require training, benefit from GPU):
   - lstm: Long Short-Term Memory network
   - gru: Gated Recurrent Unit network
   - tcn: Temporal Convolutional Network
   - nbeats: Neural Basis Expansion Analysis
   - deepar: DeepAR probabilistic forecasting
   - vanilla_transformer: Vanilla Transformer (encoder-decoder self-attention)
   - temporal_fusion_transformer: Temporal Fusion Transformer (specialized for forecasting)

Notes:
------
- All deep learning models are trained from scratch on the provided data
- No pre-trained models are required (no need to download from internet)
- Models with GPU support will automatically use GPU if available
"""

# Import prediction functions from each model
from .holt_winters import predict_holt_winters
from .prophet import predict_prophet
from .sarimax import predict_sarimax
from .xgboost import predict_xgboost
from .lstm import predict_lstm
from .gru import predict_gru
from .temporal_convolutional_network import predict_tcn
from .nbeats import predict_nbeats, predict_nbeats_interpretable
from .deepar import predict_deepar
from .vanilla_transformer import predict_transformer
from .temporal_fusion_transformer import predict_tft


# Dictionary mapping model names to prediction functions
# All functions have signature: func(train_series, horizon, random_state=None) -> pd.Series
PREDICTION_MODELS = {
    # Statistical models (fast, no GPU)
    'holt_winters': predict_holt_winters,
    'prophet': predict_prophet,
    'sarimax': predict_sarimax,
    
    # Machine learning models (fast, no GPU)
    'xgboost': predict_xgboost,
    
    # Deep learning models (train from scratch, GPU optional)
    'lstm': predict_lstm,
    'gru': predict_gru,
    'tcn': predict_tcn,
    'nbeats': predict_nbeats,
    'nbeats_interpretable': predict_nbeats_interpretable,
    'deepar': predict_deepar,
    'vanilla_transformer': predict_transformer,  # Vanilla Transformer
    'temporal_fusion_transformer': predict_tft,  # Temporal Fusion Transformer (specialized)
}


# Models that benefit from GPU acceleration
GPU_MODELS = {
    'lstm', 'gru', 'tcn', 'nbeats', 'nbeats_interpretable', 
    'deepar', 'vanilla_transformer', 'temporal_fusion_transformer'
}


# Models that are deterministic (no random_state effect)
DETERMINISTIC_MODELS = {
    'holt_winters', 'prophet', 'sarimax'
}


def get_available_models():
    """Return list of available prediction model names."""
    return list(PREDICTION_MODELS.keys())


def is_gpu_model(model_name: str) -> bool:
    """Check if model benefits from GPU acceleration."""
    return model_name in GPU_MODELS


def is_deterministic_model(model_name: str) -> bool:
    """Check if model is deterministic (random_state has no effect)."""
    return model_name in DETERMINISTIC_MODELS


__all__ = [
    'PREDICTION_MODELS',
    'GPU_MODELS',
    'DETERMINISTIC_MODELS',
    'get_available_models',
    'is_gpu_model',
    'is_deterministic_model',
    'predict_holt_winters',
    'predict_prophet',
    'predict_sarimax',
    'predict_xgboost',
    'predict_lstm',
    'predict_gru',
    'predict_tcn',
    'predict_nbeats',
    'predict_nbeats_interpretable',
    'predict_deepar',
    'predict_transformer',
    'predict_tft',
]

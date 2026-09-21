"""Prediction models used by the rolling-origin experiment."""

from .sarimax import predict_sarimax
from .xgboost import predict_xgboost

PREDICTION_MODELS = {
    "sarimax": predict_sarimax,
    "xgboost": predict_xgboost,
}
GPU_MODELS: set[str] = set()
DETERMINISTIC_MODELS = {"sarimax"}


def get_available_models():
    from framework.plugin_registry import get_prediction_models
    return list(get_prediction_models().keys())


def is_gpu_model(model_name: str) -> bool:
    if model_name in GPU_MODELS:
        return True
    from framework.plugin_registry import is_prediction_plugin_gpu
    return is_prediction_plugin_gpu(model_name)


def is_deterministic_model(model_name: str) -> bool:
    if model_name in DETERMINISTIC_MODELS:
        return True
    from framework.plugin_registry import is_prediction_plugin_deterministic
    return is_prediction_plugin_deterministic(model_name)


__all__ = [
    "PREDICTION_MODELS", "GPU_MODELS", "DETERMINISTIC_MODELS",
    "get_available_models", "is_gpu_model", "is_deterministic_model",
    "predict_sarimax", "predict_xgboost",
]

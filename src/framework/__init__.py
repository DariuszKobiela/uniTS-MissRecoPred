"""
Library API: run functions and config views (Phase A).

Requires ``src`` on ``PYTHONPATH`` (e.g. ``PYTHONPATH=src`` or editable install).
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_sr = str(_root)
if _sr not in sys.path:
    sys.path.insert(0, _sr)

from framework.config_models import (  # noqa: E402
    MetricsConfig,
    PathsConfig,
    PredictionErrorMetricsView,
    ReconstructionErrorMetricsView,
    RunConfig,
)
from framework.plugin_registry import (  # noqa: E402
    clear_plugin_registry,
    get_prediction_models,
    get_reconstruction_models,
    register_prediction_model,
    register_reconstruction_model,
)
from framework.runs import (  # noqa: E402
    PipelineFullResult,
    run_calculate_prediction_error,
    run_calculate_reconstruction_error,
    run_clean_datasets,
    run_create_split,
    run_degrade_datasets,
    run_pipeline_full,
    run_predict_datasets,
    run_reconstruct_datasets,
    run_train_prediction_models,
)

__all__ = [
    "MetricsConfig",
    "PathsConfig",
    "PipelineFullResult",
    "PredictionErrorMetricsView",
    "ReconstructionErrorMetricsView",
    "RunConfig",
    "clear_plugin_registry",
    "get_prediction_models",
    "get_reconstruction_models",
    "register_prediction_model",
    "register_reconstruction_model",
    "run_calculate_prediction_error",
    "run_calculate_reconstruction_error",
    "run_clean_datasets",
    "run_create_split",
    "run_degrade_datasets",
    "run_pipeline_full",
    "run_predict_datasets",
    "run_reconstruct_datasets",
    "run_train_prediction_models",
]

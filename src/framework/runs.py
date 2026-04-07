"""
Programmatic pipeline steps and ``run_pipeline_full``.

Implementation lives in ``src/N_*.py`` modules (loaded dynamically so numeric filenames stay valid).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

_SRC = Path(__file__).resolve().parent.parent
_sr = str(_SRC)
if _sr not in sys.path:
    sys.path.insert(0, _sr)

from utils.config_loader import (  # noqa: E402
    Config,
    PredictionModelsConfig,
    load_prediction_models_config,
)

_MOD_CACHE: dict[str, Any] = {}


def _load_script_module(filename: str) -> Any:
    if filename in _MOD_CACHE:
        return _MOD_CACHE[filename]
    path = _SRC / filename
    internal = "_pipeline_" + filename.replace(".", "_").replace("/", "_")
    spec = importlib.util.spec_from_file_location(internal, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD_CACHE[filename] = mod
    return mod


@dataclass
class PipelineFullResult:
    ok: bool
    failed_step: Optional[str] = None
    message: str = ""
    steps_completed: Optional[List[str]] = None


def run_clean_datasets(config: Config, **kwargs: Any) -> bool:
    return _load_script_module("1_clean_datasets.py").run_clean_datasets(config, **kwargs)


def run_create_split(config: Config, **kwargs: Any) -> bool:
    return _load_script_module("2_create_split.py").run_create_split(config, **kwargs)


def run_degrade_datasets(config: Config, **kwargs: Any) -> bool:
    return _load_script_module("3_degrade_datasets.py").run_degrade_datasets(config, **kwargs)


def run_reconstruct_datasets(config: Config, **kwargs: Any) -> bool:
    return _load_script_module("4_reconstruct_datasets.py").run_reconstruct_datasets(config, **kwargs)


def run_calculate_reconstruction_error(config: Config, **kwargs: Any) -> bool:
    return _load_script_module("5_calculate_reconstruction_error.py").run_calculate_reconstruction_error(
        config, **kwargs
    )


def run_train_prediction_models(
    config: Config, pred_config: PredictionModelsConfig, **kwargs: Any
) -> bool:
    return _load_script_module("7_train_prediction_models.py").run_train_prediction_models(
        config, pred_config, **kwargs
    )


def run_predict_datasets(config: Config, pred_config: PredictionModelsConfig, **kwargs: Any) -> bool:
    return _load_script_module("8_predict_datasets.py").run_predict_datasets(config, pred_config, **kwargs)


def run_calculate_prediction_error(
    config: Config, pred_config: PredictionModelsConfig, **kwargs: Any
) -> bool:
    return _load_script_module("9_calculate_prediction_error.py").run_calculate_prediction_error(
        config, pred_config, **kwargs
    )


def run_pipeline_full(
    config: Config, pred_config: Optional[PredictionModelsConfig] = None
) -> PipelineFullResult:
    """
    Same order as ``make pipeline-full``: 1, 2, 3, 4, 5, 7, 8, 9 (no Streamlit).
    """
    if pred_config is None:
        pred_config = load_prediction_models_config()

    steps: List[str] = []
    order = [
        ("clean_datasets", lambda: run_clean_datasets(config)),
        ("create_split", lambda: run_create_split(config)),
        ("degrade_datasets", lambda: run_degrade_datasets(config)),
        ("reconstruct_datasets", lambda: run_reconstruct_datasets(config)),
        ("calculate_reconstruction_error", lambda: run_calculate_reconstruction_error(config)),
        ("train_prediction_models", lambda: run_train_prediction_models(config, pred_config)),
        ("predict_datasets", lambda: run_predict_datasets(config, pred_config)),
        ("calculate_prediction_error", lambda: run_calculate_prediction_error(config, pred_config)),
    ]

    for name, fn in order:
        ok = fn()
        if not ok:
            return PipelineFullResult(
                ok=False,
                failed_step=name,
                message=f"Step {name} returned failure",
                steps_completed=steps,
            )
        steps.append(name)

    return PipelineFullResult(
        ok=True,
        failed_step=None,
        message="pipeline-full completed",
        steps_completed=steps,
    )

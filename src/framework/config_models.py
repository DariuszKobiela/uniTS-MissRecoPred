"""
Typed read-only views over nested dicts from config.yaml.

The canonical store is the full mapping held by RunConfig; these types do not copy or diverge from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

from utils.config_loader import Config


def _data_section(root: Dict[str, Any]) -> Dict[str, Any]:
    d = root.get("data") if isinstance(root, dict) else None
    return d if isinstance(d, dict) else {}


@dataclass(frozen=True)
class PathsConfig:
    """Read-only view of ``data.*`` paths (aligned with ``Config`` getters)."""

    _root: Dict[str, Any]

    @classmethod
    def from_config_dict(cls, d: Dict[str, Any]) -> "PathsConfig":
        return cls(_root=d)

    @property
    def _data(self) -> Dict[str, Any]:
        return _data_section(self._root)

    @property
    def raw_source_dir(self) -> str:
        return self._data.get("raw_source_dir", "data/0_source_data")

    @property
    def cleaned_dir(self) -> str:
        return self._data.get("cleaned_dir", "data/1_cleaned_data")

    @property
    def splitted_dir(self) -> str:
        return self._data.get("splitted_dir", "data/2_splitted_data")

    @property
    def splitted_train_dir(self) -> str:
        return self._data.get("splitted_train_dir", "data/2_splitted_data/train")

    @property
    def splitted_test_dir(self) -> str:
        return self._data.get("splitted_test_dir", "data/2_splitted_data/test")

    @property
    def source_dir(self) -> str:
        return self._data["source_dir"]

    @property
    def missing_dir(self) -> str:
        return self._data["missing_dir"]

    @property
    def fixed_dir(self) -> str:
        return self._data["fixed_dir"]

    @property
    def reconstruction_results_dir(self) -> str:
        return self._data.get("reconstruction_results_dir", "reconstruction_experiments_results")

    @property
    def prediction_results_dir(self) -> str:
        return self._data.get("prediction_results_dir", "prediction_experiment_results")


@dataclass(frozen=True)
class ReconstructionErrorMetricsView:
    _root: Dict[str, Any]

    @property
    def _em(self) -> Dict[str, Any]:
        rec = self._root.get("reconstruction", {}) or {}
        return rec.get("error_metrics", {}) or {}

    @property
    def compute(self) -> Optional[Any]:
        return self._em.get("compute")

    @property
    def primary_metric(self) -> str:
        return str(self._em.get("primary_metric", "smape")).lower()

    @property
    def primary_metric_objective(self) -> Any:
        return self._em.get("primary_metric_objective", "auto")


@dataclass(frozen=True)
class PredictionErrorMetricsView:
    _root: Dict[str, Any]

    @property
    def _em(self) -> Dict[str, Any]:
        pred = self._root.get("prediction", {}) or {}
        return pred.get("error_metrics", {}) or {}

    @property
    def compute(self) -> Optional[Any]:
        return self._em.get("compute")

    @property
    def primary_metric(self) -> str:
        return str(self._em.get("primary_metric", "mape")).lower()

    @property
    def primary_metric_lower_is_better(self) -> Optional[bool]:
        v = self._em.get("primary_metric_lower_is_better")
        if v is None:
            return None
        return bool(v)


@dataclass(frozen=True)
class MetricsConfig:
    """Views for reconstruction / prediction error metric blocks in YAML."""

    _root: Dict[str, Any]

    @classmethod
    def from_config_dict(cls, d: Dict[str, Any]) -> "MetricsConfig":
        return cls(_root=d)

    @property
    def reconstruction(self) -> ReconstructionErrorMetricsView:
        return ReconstructionErrorMetricsView(self._root)

    @property
    def prediction(self) -> PredictionErrorMetricsView:
        return PredictionErrorMetricsView(self._root)


class RunConfig:
    """
    Full config document loaded from YAML (single source of truth).

    Mutate ``.data`` before ``to_config()`` for programmatic overrides.
    """

    def __init__(self, data: Dict[str, Any], *, config_path: str = "") -> None:
        self.config_path = config_path
        self._data = data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
        return cls(raw, config_path=path)

    @property
    def paths(self) -> PathsConfig:
        return PathsConfig.from_config_dict(self._data)

    @property
    def metrics(self) -> MetricsConfig:
        return MetricsConfig.from_config_dict(self._data)

    def to_config(self) -> Config:
        return Config.from_dict(self._data, self.config_path)

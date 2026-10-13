#!/usr/bin/env python3
"""Merge the latest complete reconstruction and rolling-origin result batches."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from export_prediction_parquet_by_model import export_by_model

ROOT = Path(__file__).resolve().parents[1]

RECONSTRUCTION_INPUTS = [
    "reconstruction_experiments_results/reconstruction_results_20260927_230400.csv",
    "reconstruction_experiments_results/reconstruction_results_20261007_131753.csv",
    "reconstruction_experiments_results/reconstruction_results_20261007_131925.csv",
    "reconstruction_experiments_results/reconstruction_results_20261007_165940.csv",
    "reconstruction_experiments_results/reconstruction_results_20261008_125556.csv",
    "reconstruction_experiments_results/reconstruction_results_20261010_170722.csv",
]

ROLLING_INPUTS = [
    "prediction_experiment_results/prediction_results_rolling_origins_20260929_151017.csv",
    "prediction_experiment_results/prediction_results_rolling_origins_20261001_114108.csv",
    "prediction_experiment_results/prediction_results_rolling_origins_20261007_121603.csv",
    "prediction_experiment_results/prediction_results_rolling_origins_20261007_161856.csv",
    "prediction_experiment_results/prediction_results_rolling_origins_20261008_111344.csv",
    "prediction_experiment_results/prediction_results_rolling_origins_20261010_153028.csv",
    # Complete Prophet batch for all 17 reconstruction methods and originals.
    "prediction_experiment_results/prediction_results_rolling_origins_20261010_194434.csv",
    # Complete Holt-Winters batch for all 17 reconstruction methods and originals.
    "prediction_experiment_results/prediction_results_rolling_origins_20261010_201923.csv",
]

RECONSTRUCTION_KEY = [
    "dataset_name",
    "technique",
    "structure",
    "rate_percent",
    "iteration",
    "model",
]

ROLLING_KEY = [
    "dataset_name",
    "source_type",
    "technique",
    "structure",
    "rate_percent",
    "reconstruction_iteration",
    "reconstruction_model",
    "prediction_model",
    "evaluation_scheme",
    "origin",
    "forecast_horizon",
    "metric_scope",
    "lead_time_start",
    "lead_time_end",
]


def read_batches(
    paths: list[str],
    *,
    superseded_first_model_column: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    for index, relative_path in enumerate(paths):
        path = ROOT / relative_path
        frame = pd.read_csv(path, low_memory=False)
        rows_read = len(frame)
        if superseded_first_model_column and index == 0:
            # The original classical run had two failed SARIMAX/XGBoost tasks.
            # A later complete SARIMAX run supersedes this model's rows.
            frame = frame[frame[superseded_first_model_column] != "sarimax"].copy()
        frames.append(frame)
        sources.append(
            {
                "path": relative_path,
                "rows_read": rows_read,
                "rows_selected": len(frame),
            }
        )
    return pd.concat(frames, ignore_index=True), sources


def assert_unique(frame: pd.DataFrame, key: list[str], label: str) -> None:
    duplicate_count = int(frame.duplicated(key, keep=False).sum())
    if duplicate_count:
        raise ValueError(f"{label}: found {duplicate_count} rows with duplicate result keys")


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    reconstruction, reconstruction_sources = read_batches(
        RECONSTRUCTION_INPUTS,
        superseded_first_model_column="model",
    )
    assert_unique(reconstruction, RECONSTRUCTION_KEY, "reconstruction")
    reconstruction_path = (
        ROOT
        / "reconstruction_experiments_results"
        / f"reconstruction_results_merged_{stamp}.csv"
    )
    reconstruction.to_csv(reconstruction_path, index=False)

    rolling, rolling_sources = read_batches(
        ROLLING_INPUTS,
        superseded_first_model_column="reconstruction_model",
    )
    assert_unique(rolling, ROLLING_KEY, "rolling-origin")
    rolling_path = (
        ROOT
        / "prediction_experiment_results"
        / f"prediction_results_rolling_origins_merged_{stamp}.csv"
    )
    rolling.to_csv(rolling_path, index=False)
    parquet_manifest = export_by_model(
        rolling_path,
        ROOT / "prediction_experiment_results" / "latest_by_model",
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "latest complete batch per reconstruction model; old files preserved",
        "reconstruction": {
            "output": str(reconstruction_path.relative_to(ROOT)),
            "rows": len(reconstruction),
            "models": reconstruction["model"].value_counts().sort_index().to_dict(),
            "sources": reconstruction_sources,
        },
        "rolling_origin": {
            "output": str(rolling_path.relative_to(ROOT)),
            "rows": len(rolling),
            "prediction_models": (
                rolling["prediction_model"].value_counts().sort_index().to_dict()
            ),
            "reconstruction_models": (
                rolling["reconstruction_model"]
                .fillna("<baseline>")
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "sources": rolling_sources,
            "parquet_manifest": str(parquet_manifest.relative_to(ROOT)),
        },
    }
    manifest_path = ROOT / f"merged_results_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(reconstruction_path)
    print(rolling_path)
    print(manifest_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch statistical export for rebuttal rolling-origin and reconstruction results."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import load_config
from utils.progress import tqdm
from utils.statistical_tests import friedman_test, pairwise_comparisons

ROLLING_PAIR_COLUMNS = [
    "dataset_name",
    "technique",
    "structure",
    "rate_percent",
    "reconstruction_iteration",
    "reconstruction_model",
    "source_type",
    "forecast_horizon",
    "metric_scope",
    "origin",
]

RECONSTRUCTION_PAIR_COLUMNS = [
    "dataset_name",
    "technique",
    "structure",
    "rate_percent",
    "iteration",
]


def _prepare_rolling_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["model"] = prepared["prediction_model"]
    if "reconstruction_iteration" not in prepared.columns and "iteration" in prepared.columns:
        prepared["reconstruction_iteration"] = prepared["iteration"]
    if metric not in prepared.columns:
        raise ValueError(f"Metric column {metric!r} missing from rolling-origin results")
    cumulative = prepared[prepared.get("metric_scope", "cumulative") == "cumulative"].copy()
    if cumulative.empty:
        cumulative = prepared
    return cumulative


def _prepare_reconstruction_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    prepared = frame.copy()
    if "model" not in prepared.columns and "reconstruction_model" in prepared.columns:
        prepared["model"] = prepared["reconstruction_model"]
    if metric not in prepared.columns:
        raise ValueError(f"Metric column {metric!r} missing from reconstruction results")
    return prepared


def export_statistics(
    frame: pd.DataFrame,
    *,
    metric: str,
    pair_columns: list[str],
    output_dir: Path,
    prefix: str,
    lower_is_better: bool = True,
) -> dict:
    pairwise = pairwise_comparisons(
        frame,
        metric=metric,
        pair_columns=pair_columns,
        lower_is_better=lower_is_better,
        show_progress=True,
        progress_desc=f"{prefix}: paired tests",
    )
    omnibus = friedman_test(frame, metric=metric, pair_columns=pair_columns)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pairwise_path = output_dir / f"{prefix}_pairwise_{metric}_{timestamp}.csv"
    omnibus_path = output_dir / f"{prefix}_friedman_{metric}_{timestamp}.json"
    pairwise.to_csv(pairwise_path, index=False)
    omnibus_path.write_text(json.dumps(omnibus, indent=2), encoding="utf-8")
    return {
        "metric": metric,
        "pairwise_file": str(pairwise_path),
        "friedman_file": str(omnibus_path),
        "omnibus": omnibus,
        "n_pairwise_rows": len(pairwise),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export batch Friedman and paired post-hoc statistics")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--rolling-results",
        help="CSV from src/12_evaluate_rolling_origins.py",
    )
    parser.add_argument(
        "--reconstruction-results",
        help="CSV from src/11_calculate_reconstruction_error.py",
    )
    parser.add_argument("--output-dir", default="prediction_experiment_results/batch_statistics")
    parser.add_argument("--metric", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_metric = args.metric or config.get_prediction_primary_metric()
    reconstruction_metric = config.get_reconstruction_primary_metric()

    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exports": [],
    }

    if not args.rolling_results and not args.reconstruction_results:
        raise ValueError("Provide --rolling-results and/or --reconstruction-results")

    n_exports = int(bool(args.rolling_results)) + int(bool(args.reconstruction_results))
    export_progress = tqdm(total=n_exports, desc="Batch statistics", unit="export")

    if args.rolling_results:
        rolling = pd.read_csv(args.rolling_results)
        prepared = _prepare_rolling_frame(rolling, forecast_metric)
        summary["exports"].append(
            export_statistics(
                prepared,
                metric=forecast_metric,
                pair_columns=ROLLING_PAIR_COLUMNS,
                output_dir=output_dir,
                prefix="rolling_origin",
                lower_is_better=True,
            )
        )
        export_progress.update()

    if args.reconstruction_results:
        reconstruction = pd.read_csv(args.reconstruction_results)
        prepared = _prepare_reconstruction_frame(reconstruction, reconstruction_metric)
        summary["exports"].append(
            export_statistics(
                prepared,
                metric=reconstruction_metric,
                pair_columns=RECONSTRUCTION_PAIR_COLUMNS,
                output_dir=output_dir,
                prefix="reconstruction",
                lower_is_better=True,
            )
        )
        export_progress.update()

    export_progress.close()

    summary_path = output_dir / f"batch_statistics_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote batch statistics summary to {summary_path}")


if __name__ == "__main__":
    main()

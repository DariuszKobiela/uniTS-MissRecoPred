#!/usr/bin/env python3
"""Create one JSON summary of the current experiment outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def newest(pattern: str) -> Path | None:
    files = list(Path().glob(pattern))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def csv_info(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    frame = pd.read_csv(path)
    return {"path": str(path), "rows": len(frame), "columns": list(frame.columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize current experiment outputs")
    parser.add_argument("--output", default="experiment_run_summary.json")
    args = parser.parse_args()
    split_manifest = Path("data/2_splitted_data/split_manifest.json")
    training_report = Path("models/sd2_windowed/training_report.json")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "inputs": sorted(str(path) for path in Path("data/0_source_data").glob("*.csv")),
        "cleaning_report": csv_info(Path("data/1_cleaned_data/reports/cleaning_report.csv")),
        "split_manifest": str(split_manifest) if split_manifest.is_file() else None,
        "sd2_training_report": str(training_report) if training_report.is_file() else None,
        "missingness": csv_info(Path("data/3_missing_data/reports/missingness_realizations.csv")),
        "reconstruction": csv_info(newest("reconstruction_experiments_results/reconstruction_results_*.csv")),
        "rolling_origin": csv_info(newest("prediction_experiment_results/prediction_results_rolling_origins_*.csv")),
        "batch_statistics": sorted(str(path) for path in Path("prediction_experiment_results/batch_statistics").glob("*")),
    }
    required = [summary["split_manifest"], summary["missingness"], summary["reconstruction"], summary["rolling_origin"]]
    if not all(required):
        summary["status"] = "incomplete"
    target = Path(args.output)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Zapisano podsumowanie runu: {target}")
    print(f"Status: {summary['status']}")


if __name__ == "__main__":
    main()

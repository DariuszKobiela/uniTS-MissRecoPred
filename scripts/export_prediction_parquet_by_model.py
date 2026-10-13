#!/usr/bin/env python3
"""Export the latest merged rolling-origin CSV as one Parquet file per model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "prediction_experiment_results"
STRING_COLUMNS = {
    "forecast_fallback_reason": "string",
    "forecast_requested_order": "string",
    "forecast_requested_seasonal_order": "string",
    "forecast_fitted_order": "string",
    "forecast_fitted_seasonal_order": "string",
}


def latest_merged_csv(results_dir: Path) -> Path:
    candidates = sorted(
        results_dir.glob("prediction_results_rolling_origins_merged_*.csv"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No merged rolling-origin CSV found in {results_dir}")
    return candidates[0]


def export_by_model(
    input_path: Path,
    output_dir: Path,
    *,
    chunksize: int = 100_000,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    temporary_paths: dict[str, Path] = {}
    row_counts: dict[str, int] = {}

    for stale_temporary in output_dir.glob(".*.parquet.tmp"):
        stale_temporary.unlink()

    try:
        for chunk in pd.read_csv(
            input_path,
            low_memory=False,
            chunksize=chunksize,
            dtype=STRING_COLUMNS,
        ):
            if "prediction_model" not in chunk.columns:
                raise ValueError("Input has no prediction_model column")
            for model_name, model_frame in chunk.groupby("prediction_model", sort=False):
                model = str(model_name)
                table = pa.Table.from_pandas(model_frame.reset_index(drop=True), preserve_index=False)
                if model not in writers:
                    temporary = output_dir / f".{model}.parquet.tmp"
                    temporary_paths[model] = temporary
                    writers[model] = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                    row_counts[model] = 0
                writers[model].write_table(table)
                row_counts[model] += len(model_frame)
    except Exception:
        for writer in writers.values():
            writer.close()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise
    else:
        for writer in writers.values():
            writer.close()

    outputs: dict[str, dict[str, int | str]] = {}
    for model, temporary in temporary_paths.items():
        destination = output_dir / f"{model}.parquet"
        temporary.replace(destination)
        outputs[model] = {
            "path": str(destination.relative_to(ROOT)),
            "rows": row_counts[model],
            "bytes": destination.stat().st_size,
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": str(input_path.relative_to(ROOT)),
                "format": "parquet",
                "compression": "zstd",
                "models": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prediction_experiment_results/latest_by_model"),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    input_path = args.input or latest_merged_csv(DEFAULT_RESULTS_DIR)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest = export_by_model(input_path, output_dir, chunksize=args.chunksize)
    print(manifest)


if __name__ == "__main__":
    main()

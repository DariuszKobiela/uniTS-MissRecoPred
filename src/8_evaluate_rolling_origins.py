#!/usr/bin/env python3
"""Evaluate SARIMAX and local XGBoost with rolling origins inside the test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from prediction_metrics import compute_prediction_metrics
from prediction_models.sarimax import predict_sarimax
from utils.config_loader import load_config, load_prediction_models_config
from utils.experiment_naming import decode_missingness_label, encode_missingness_label
from utils.rolling_origins import (
    expanding_history,
    fit_predict_xgboost_local,
    plan_rolling_origins,
)

SUPPORTED_MODELS = {"sarimax", "xgboost"}


def load_complete_series(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").reset_index(drop=True)
    if series.empty:
        raise ValueError(f"Empty series: {path}")
    if series.isna().any():
        raise ValueError(f"Series still contains missing values: {path}")
    return series.astype(float)


def parse_reconstructed_filename(filename: str) -> dict[str, Any]:
    base_name = Path(filename).stem
    parts = base_name.split("_")
    rate_idx = next(
        (index for index, part in enumerate(parts) if part.endswith("p") and part[:-1].isdigit()),
        None,
    )
    if rate_idx is None or rate_idx < 1 or rate_idx + 2 >= len(parts):
        raise ValueError(f"Invalid reconstructed filename: {filename}")
    technique, structure = decode_missingness_label(parts[rate_idx - 1])
    return {
        "dataset_name": "_".join(parts[: rate_idx - 1]),
        "source_type": "reconstructed",
        "technique": technique,
        "structure": structure,
        "rate_percent": int(parts[rate_idx][:-1]),
        "reconstruction_iteration": int(parts[rate_idx + 1]),
        "reconstruction_model": "_".join(parts[rate_idx + 2 :]),
    }


def discover_sources(config) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if config.get_predict_on_original_train():
        for path in sorted(Path(config.get_splitted_train_dir()).glob("*.csv")):
            sources.append(
                {
                    "path": str(path),
                    "dataset_name": path.stem,
                    "source_type": "original",
                    "technique": None,
                    "structure": None,
                    "rate_percent": None,
                    "reconstruction_iteration": None,
                    "reconstruction_model": None,
                }
            )

    if config.get_predict_on_reconstructed():
        for path in sorted(Path(config.get_fixed_dir()).glob("*.csv")):
            try:
                metadata = parse_reconstructed_filename(path.name)
            except ValueError:
                continue
            sources.append({"path": str(path), **metadata})
    return sources


def resolve_horizons(config, dataset: str, test_length: int) -> list[int]:
    configured = config.get_experiment_horizons()
    horizons = None
    for key, values in configured.items():
        if Path(key).stem == dataset:
            horizons = values
            break
    if horizons is None:
        horizons = [test_length]
    valid = sorted({int(value) for value in horizons if 0 < int(value) <= test_length})
    if not valid:
        raise ValueError(f"No configured horizon fits test length {test_length} for {dataset}")
    return valid


def source_label(metadata: dict[str, Any]) -> str:
    if metadata["source_type"] == "original":
        return f"{metadata['dataset_name']}_original"
    label = encode_missingness_label(metadata['technique'], metadata['structure'])
    return (
        f"{metadata['dataset_name']}_{label}_"
        f"{metadata['rate_percent']}p_{metadata['reconstruction_iteration']}_"
        f"{metadata['reconstruction_model']}"
    )


def forecast_at_origin(
    model_name: str,
    history: pd.Series,
    horizon: int,
    model_params: dict,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "xgboost":
        params = dict(model_params)
        params["n_jobs"] = 1
        return fit_predict_xgboost_local(history, horizon, params, seed), {}
    if model_name == "sarimax":
        params = dict(model_params)
        if "order" in params:
            params["order"] = tuple(params["order"])
        if "seasonal_order" in params:
            params["seasonal_order"] = tuple(params["seasonal_order"])
        forecast = predict_sarimax(history, horizon, **params)
        return np.asarray(forecast, dtype=np.float64), dict(forecast.attrs)
    raise ValueError(f"Unsupported rolling-origin model: {model_name}")


def save_origin_prediction(
    output_dir: Path,
    metadata: dict[str, Any],
    model_name: str,
    origin,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> str:
    filename = (
        f"{source_label(metadata)}_{model_name}_h{origin.horizon}_origin{origin.number}_offset{origin.test_offset}.csv"
    )
    path = output_dir / filename
    frame = pd.DataFrame(
        {
            "test_position": np.arange(origin.test_offset, origin.test_stop),
            "actual": actual,
            "predicted": predicted,
            "forecast_horizon": origin.horizon,
            "origin": origin.number,
            "origin_offset": origin.test_offset,
        }
    )
    frame.to_csv(path, index=False)
    return str(path)


def evaluate_source_model(task: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = task["metadata"]
    model_name = task["model_name"]
    train = load_complete_series(metadata["path"])
    test = load_complete_series(task["test_path"])
    result_metadata = {key: value for key, value in metadata.items() if key != "path"}
    results: list[dict[str, Any]] = []

    for horizon in task["horizons"]:
        origins = plan_rolling_origins(len(test), horizon, task["n_origins"])
        for origin in origins:
            history = expanding_history(train, test, origin.test_offset)
            actual = test.iloc[origin.test_offset : origin.test_stop].to_numpy(dtype=np.float64)
            started = time.perf_counter()
            predicted, forecast_metadata = forecast_at_origin(
                model_name,
                history,
                horizon,
                task["model_params"],
                task["seed"] + origin.number,
            )
            elapsed = time.perf_counter() - started
            if len(predicted) != horizon or not np.isfinite(predicted).all():
                raise ValueError(f"{model_name} returned an invalid forecast at origin {origin.number}")

            metrics = compute_prediction_metrics(
                actual,
                predicted,
                train=history.to_numpy(dtype=np.float64),
                metric_keys=task["metric_keys"],
            )
            abs_error = np.abs(actual - predicted)
            prediction_file = save_origin_prediction(
                Path(task["predictions_dir"]),
                metadata,
                model_name,
                origin,
                actual,
                predicted,
            )
            results.append(
                {
                    **result_metadata,
                    "prediction_model": model_name,
                    "prediction_iteration": 1,
                    "evaluation_scheme": "rolling_origin_test_expanding",
                    "forecast_horizon": horizon,
                    "origin": origin.number,
                    "origin_count_for_horizon": len(origins),
                    "origin_offset": origin.test_offset,
                    "test_start": origin.test_offset,
                    "test_stop_exclusive": origin.test_stop,
                    "revealed_test_samples": origin.test_offset,
                    "base_train_samples": len(train),
                    "effective_train_samples": len(history),
                    "forecast_fallback_used": forecast_metadata.get("fallback_used", False),
                    "forecast_fallback_type": forecast_metadata.get("fallback_type", "none"),
                    "forecast_fallback_reason": forecast_metadata.get("fallback_reason", ""),
                    "forecast_requested_order": forecast_metadata.get("requested_order", ""),
                    "forecast_requested_seasonal_order": forecast_metadata.get(
                        "requested_seasonal_order", ""
                    ),
                    "forecast_fitted_order": forecast_metadata.get("fitted_order", ""),
                    "forecast_fitted_seasonal_order": forecast_metadata.get(
                        "fitted_seasonal_order", ""
                    ),
                    **metrics,
                    "max_error": float(abs_error.max()),
                    "min_error": float(abs_error.min()),
                    "std_error": float(abs_error.std(ddof=0)),
                    "n_samples": horizon,
                    "time_seconds": elapsed,
                    "prediction_file": prediction_file,
                }
            )
    return results


def aggregate_origin_results(frame: pd.DataFrame, metric_keys: list[str]) -> pd.DataFrame:
    """Aggregate metrics across origins without mixing experimental conditions."""
    group_columns = [
        "dataset_name",
        "source_type",
        "technique",
        "rate_percent",
        "reconstruction_iteration",
        "reconstruction_model",
        "prediction_model",
        "forecast_horizon",
        "evaluation_scheme",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "origins_evaluated": ("origin", "nunique"),
        "time_seconds_mean": ("time_seconds", "mean"),
        "time_seconds_sum": ("time_seconds", "sum"),
    }
    for metric in metric_keys:
        if metric in frame.columns:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_median"] = (metric, "median")
            aggregations[f"{metric}_std"] = (metric, "std")
    return frame.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()


def run(config, pred_config, models: list[str] | None = None, n_origins: int | None = None) -> Path:
    settings = config.config.get("prediction", {}).get("rolling_origins", {}) or {}
    selected = models or settings.get("models", ["sarimax", "xgboost"])
    selected = [str(model).lower() for model in selected]
    unknown = set(selected) - SUPPORTED_MODELS
    if unknown:
        raise ValueError(f"Unsupported rolling-origin models: {sorted(unknown)}")
    if not selected:
        raise ValueError("No rolling-origin models selected")

    requested_origins = int(n_origins if n_origins is not None else settings.get("n_origins", 5))
    if requested_origins < 1:
        raise ValueError("n_origins must be positive")

    test_paths = {path.stem: str(path) for path in Path(config.get_splitted_test_dir()).glob("*.csv")}
    sources = discover_sources(config)
    if not sources:
        raise FileNotFoundError("No original or reconstructed training series found")

    output_root = Path(config.get_prediction_results_dir()) / "rolling_origins"
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = config.get_prediction_error_metrics_to_compute()

    tasks = []
    skipped = []
    selected_datasets = {Path(str(value)).stem for value in settings.get("datasets", []) if str(value).strip()}
    for metadata in sources:
        dataset = metadata["dataset_name"]
        test_path = test_paths.get(dataset)
        if selected_datasets and dataset not in selected_datasets:
            continue
        if test_path is None:
            skipped.append(dataset)
            continue
        test_length = len(load_complete_series(test_path))
        horizons = resolve_horizons(config, dataset, test_length)
        for model_name in selected:
            params = pred_config.get_model_params(model_name)
            tasks.append(
                {
                    "metadata": metadata,
                    "model_name": model_name,
                    "model_params": params,
                    "test_path": test_path,
                    "horizons": horizons,
                    "n_origins": requested_origins,
                    "metric_keys": metric_keys,
                    "predictions_dir": str(predictions_dir),
                    "seed": int(settings.get("seed", 42)),
                }
            )

    if not tasks:
        raise ValueError("No rolling-origin tasks could be constructed")
    if skipped:
        print(f"Skipped sources without matching test set: {len(skipped)}")

    configured_workers = int(settings.get("max_workers", 1))
    max_workers = max(1, configured_workers)
    print("=" * 72)
    print("ROLLING-ORIGIN TEST EVALUATION")
    print("=" * 72)
    print(f"Models: {selected}")
    print(f"Requested origins per horizon: {requested_origins}")
    print(f"Tasks (source x model): {len(tasks)}")
    print(f"Workers: {max_workers}")
    print("Later origins refit on train + revealed real test prefix.")
    print("=" * 72)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if max_workers == 1:
        for task in tqdm(tasks, desc="Rolling origins", unit="task"):
            try:
                results.extend(evaluate_source_model(task))
            except Exception as exc:
                errors.append(
                    {
                        "source": task["metadata"]["path"],
                        "model": task["model_name"],
                        "error": str(exc),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(evaluate_source_model, task): task for task in tasks}
            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Rolling origins",
                unit="task",
            ):
                task = future_to_task[future]
                try:
                    results.extend(future.result())
                except Exception as exc:
                    errors.append(
                        {
                            "source": task["metadata"]["path"],
                            "model": task["model_name"],
                            "error": str(exc),
                        }
                    )

    if not results:
        raise RuntimeError(f"All rolling-origin tasks failed: {errors[:5]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(config.get_prediction_results_dir()) / (f"prediction_results_rolling_origins_{timestamp}.csv")
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_path, index=False)
    aggregate_path = output_root / f"origin_summary_{timestamp}.csv"
    aggregate_origin_results(result_frame, metric_keys).to_csv(aggregate_path, index=False)

    summary_path = output_root / f"run_summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "models": selected,
                "requested_origins": requested_origins,
                "tasks": len(tasks),
                "successful_forecasts": len(results),
                "errors": errors,
                "result_file": str(output_path),
                "aggregate_result_file": str(aggregate_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} origin-level results to {output_path}")
    print(f"Saved across-origin summary to {aggregate_path}")
    print(f"Errors: {len(errors)}; summary: {summary_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models", nargs="+", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--n-origins", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    pred_config = load_prediction_models_config()
    run(config, pred_config, models=args.models, n_origins=args.n_origins)


if __name__ == "__main__":
    main()

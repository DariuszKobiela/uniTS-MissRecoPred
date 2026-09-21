#!/usr/bin/env python3
"""Authoritative rolling-origin forecast evaluation for the rebuttal protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from prediction_models.sarimax import predict_sarimax
from utils.config_loader import load_config, load_prediction_models_config
from utils.experiment_naming import decode_missingness_label, encode_missingness_label
from utils.rolling_origins import (
    evaluate_forecast_slices,
    expanding_history,
    fit_predict_xgboost_direct_missing,
    fit_predict_xgboost_local,
    plan_rolling_origins,
    predict_persistence,
    predict_seasonal_naive,
    resolve_seasonal_period,
)
from utils.split_plan import resolve_n_origins

SUPPORTED_MODELS = {
    "sarimax",
    "xgboost",
    "persistence",
    "seasonal_naive",
    "xgboost_direct_missing",
}


def load_complete_series(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").reset_index(drop=True)
    if series.empty:
        raise ValueError(f"Empty series: {path}")
    if series.isna().any():
        raise ValueError(f"Series still contains missing values: {path}")
    return series.astype(float)


def load_series_allow_missing(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce").reset_index(drop=True).astype(float)


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
                    "allow_missing_history": False,
                }
            )

    if config.get_predict_on_reconstructed():
        for path in sorted(Path(config.get_fixed_dir()).glob("*.csv")):
            try:
                metadata = parse_reconstructed_filename(path.name)
            except ValueError:
                continue
            sources.append(
                {
                    "path": str(path),
                    **metadata,
                    "allow_missing_history": False,
                }
            )

    if config.get_rolling_origin_settings().get("include_direct_missing", True):
        for path in sorted(Path(config.get_missing_dir()).glob("*.csv")):
            stem = path.stem
            rate_idx = next(
                (index for index, part in enumerate(stem.split("_")) if part.endswith("p") and part[:-1].isdigit()),
                None,
            )
            if rate_idx is None or rate_idx < 1:
                continue
            technique, structure = decode_missingness_label(stem.split("_")[rate_idx - 1])
            dataset_name = "_".join(stem.split("_")[: rate_idx - 1])
            sources.append(
                {
                    "path": str(path),
                    "dataset_name": dataset_name,
                    "source_type": "degraded",
                    "technique": technique,
                    "structure": structure,
                    "rate_percent": int(stem.split("_")[rate_idx][:-1]),
                    "reconstruction_iteration": int(stem.split("_")[rate_idx + 1]),
                    "reconstruction_model": None,
                    "allow_missing_history": True,
                }
            )
    return sources


def resolve_horizons(config, dataset: str, test_length: int) -> list[int]:
    configured = config.get_experiment_horizons()
    horizons = None
    for key, values in configured.items():
        if Path(key).stem == dataset or key == dataset:
            horizons = values
            break
    if horizons is None:
        horizons = [test_length]
    valid = sorted({int(value) for value in horizons if 0 < int(value) <= test_length})
    if not valid:
        raise ValueError(f"No configured horizon fits test length {test_length} for {dataset}")
    return valid


def source_label(metadata: dict[str, Any], model_name: str | None = None) -> str:
    if metadata["source_type"] == "original":
        label = f"{metadata['dataset_name']}_original"
    elif metadata["source_type"] == "degraded":
        missingness = encode_missingness_label(metadata["technique"], metadata["structure"])
        label = (
            f"{metadata['dataset_name']}_{missingness}_"
            f"{metadata['rate_percent']}p_{metadata['reconstruction_iteration']}_direct_missing"
        )
    else:
        missingness = encode_missingness_label(metadata["technique"], metadata["structure"])
        label = (
            f"{metadata['dataset_name']}_{missingness}_"
            f"{metadata['rate_percent']}p_{metadata['reconstruction_iteration']}_"
            f"{metadata['reconstruction_model']}"
        )
    if model_name == "xgboost_direct_missing" and metadata["source_type"] != "degraded":
        return f"{label}_direct_missing_view"
    return label


def forecast_at_origin(
    model_name: str,
    history: pd.Series,
    horizon: int,
    model_params: dict,
    seed: int,
    *,
    seasonal_period: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {"fallback_used": False, "fallback_type": "none", "fallback_reason": ""}
    if model_name == "xgboost":
        params = dict(model_params)
        params["n_jobs"] = 1
        return fit_predict_xgboost_local(history, horizon, params, seed), metadata
    if model_name == "xgboost_direct_missing":
        params = dict(model_params)
        params["n_jobs"] = 1
        return fit_predict_xgboost_direct_missing(history, horizon, params, seed), metadata
    if model_name == "persistence":
        return predict_persistence(history, horizon), metadata
    if model_name == "seasonal_naive":
        return predict_seasonal_naive(history, horizon, seasonal_period), metadata
    if model_name == "sarimax":
        params = dict(model_params)
        if "order" in params:
            params["order"] = tuple(params["order"])
        if "seasonal_order" in params:
            params["seasonal_order"] = tuple(params["seasonal_order"])
        forecast = predict_sarimax(history, horizon, **params)
        attrs = dict(forecast.attrs)
        metadata.update(
            {
                "fallback_used": attrs.get("fallback_used", False),
                "fallback_type": attrs.get("fallback_type", "none"),
                "fallback_reason": attrs.get("fallback_reason", ""),
                "requested_order": attrs.get("requested_order", ""),
                "requested_seasonal_order": attrs.get("requested_seasonal_order", ""),
                "fitted_order": attrs.get("fitted_order", ""),
                "fitted_seasonal_order": attrs.get("fitted_seasonal_order", ""),
            }
        )
        return np.asarray(forecast, dtype=np.float64), metadata
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
        f"{source_label(metadata, model_name)}_{model_name}_hmax{origin.horizon}_"
        f"origin{origin.number}_offset{origin.test_offset}.csv"
    )
    path = output_dir / filename
    frame = pd.DataFrame(
        {
            "test_position": np.arange(origin.test_offset, origin.test_offset + len(actual)),
            "actual": actual,
            "predicted": predicted,
            "forecast_horizon_max": origin.horizon,
            "origin": origin.number,
            "origin_offset": origin.test_offset,
        }
    )
    frame.to_csv(path, index=False)
    return str(path)


def evaluate_source_model(task: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = task["metadata"]
    model_name = task["model_name"]
    allow_missing = metadata.get("allow_missing_history", False)

    if model_name == "xgboost_direct_missing":
        if metadata["source_type"] != "degraded":
            return []
        train = load_series_allow_missing(metadata["path"])
    elif allow_missing:
        train = load_series_allow_missing(metadata["path"])
    else:
        train = load_complete_series(metadata["path"])

    test = load_complete_series(task["test_path"])
    result_metadata = {key: value for key, value in metadata.items() if key != "path"}
    results: list[dict[str, Any]] = []

    h_max = max(task["horizons"])
    origins = plan_rolling_origins(len(test), h_max, task["n_origins"])

    for origin in origins:
        history = expanding_history(
            train,
            test,
            origin.test_offset,
            allow_missing=allow_missing or model_name == "xgboost_direct_missing",
        )
        actual = test.iloc[origin.test_offset : origin.test_offset + h_max].to_numpy(dtype=np.float64)
        started = time.perf_counter()
        predicted, forecast_metadata = forecast_at_origin(
            model_name,
            history,
            h_max,
            task["model_params"],
            task["seed"] + origin.number,
            seasonal_period=task["seasonal_period"],
        )
        elapsed = time.perf_counter() - started
        if len(predicted) != h_max or not np.isfinite(predicted).all():
            raise ValueError(
                f"{model_name} returned invalid H_max={h_max} forecast at origin {origin.number}"
            )

        train_history = history.to_numpy(dtype=np.float64)
        train_history = train_history[np.isfinite(train_history)]
        metric_rows = evaluate_forecast_slices(
            actual,
            predicted,
            task["horizons"],
            train_history,
            task["metric_keys"],
            include_lead_time_bins=task["include_lead_time_bins"],
        )
        prediction_file = save_origin_prediction(
            Path(task["predictions_dir"]),
            metadata,
            model_name,
            origin,
            actual,
            predicted,
        )

        for metric_row in metric_rows:
            abs_error = np.abs(actual[: metric_row["forecast_horizon"]] - predicted[: metric_row["forecast_horizon"]])
            results.append(
                {
                    **result_metadata,
                    "prediction_model": model_name,
                    "prediction_iteration": 1,
                    "evaluation_scheme": "rolling_origin_test_expanding_hmax",
                    "origin": origin.number,
                    "origin_count_for_hmax": len(origins),
                    "origin_offset": origin.test_offset,
                    "test_start": origin.test_offset,
                    "test_stop_exclusive": origin.test_offset + h_max,
                    "revealed_test_samples": origin.test_offset,
                    "base_train_samples": len(train),
                    "effective_train_samples": len(history),
                    "forecast_horizon_max": h_max,
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
                    **metric_row,
                    "max_error": float(abs_error.max()) if len(abs_error) else np.nan,
                    "min_error": float(abs_error.min()) if len(abs_error) else np.nan,
                    "std_error": float(abs_error.std(ddof=0)) if len(abs_error) else np.nan,
                    "time_seconds": elapsed,
                    "prediction_file": prediction_file,
                }
            )
    return results


def aggregate_origin_results(frame: pd.DataFrame, metric_keys: list[str]) -> pd.DataFrame:
    group_columns = [
        "dataset_name",
        "source_type",
        "technique",
        "structure",
        "rate_percent",
        "reconstruction_iteration",
        "reconstruction_model",
        "prediction_model",
        "forecast_horizon",
        "metric_scope",
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
    present = [column for column in group_columns if column in frame.columns]
    return frame.groupby(present, dropna=False).agg(**aggregations).reset_index()


def run(config, pred_config, models: list[str] | None = None, n_origins: int | None = None) -> Path:
    settings = config.get_rolling_origin_settings()
    selected = models or settings.get("models", list(SUPPORTED_MODELS))
    selected = [str(model).lower() for model in selected]
    unknown = set(selected) - SUPPORTED_MODELS
    if unknown:
        raise ValueError(f"Unsupported rolling-origin models: {sorted(unknown)}")
    if not selected:
        raise ValueError("No rolling-origin models selected")

    default_origins = int(n_origins if n_origins is not None else settings.get("n_origins", 5))
    origin_counts = config.get_rolling_origin_counts()
    seasonal_periods = config.get_seasonal_periods()

    test_paths = {path.stem: str(path) for path in Path(config.get_splitted_test_dir()).glob("*.csv")}
    sources = discover_sources(config)
    if not sources:
        raise FileNotFoundError("No original, reconstructed, or degraded training series found")

    output_root = Path(config.get_prediction_results_dir()) / "rolling_origins"
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = config.get_prediction_error_metrics_to_compute()
    include_lead_time_bins = bool(settings.get("lead_time_bins", True))

    tasks = []
    skipped = []
    selected_datasets = {
        Path(str(value)).stem for value in settings.get("datasets", []) if str(value).strip()
    }
    for metadata in sources:
        dataset = metadata["dataset_name"]
        test_path = test_paths.get(dataset)
        if selected_datasets and dataset not in selected_datasets:
            continue
        if test_path is None:
            skipped.append(dataset)
            continue

        if metadata["source_type"] == "degraded":
            if "xgboost_direct_missing" not in selected:
                continue
            model_subset = ["xgboost_direct_missing"]
        else:
            model_subset = [m for m in selected if m != "xgboost_direct_missing"]
            if not model_subset:
                continue

        test_length = len(load_complete_series(test_path))
        horizons = resolve_horizons(config, dataset, test_length)
        dataset_origins = resolve_n_origins(dataset, origin_counts, fallback=default_origins)
        seasonal_period = resolve_seasonal_period(dataset, seasonal_periods)

        for model_name in model_subset:
            params = pred_config.get_model_params(
                "xgboost" if model_name == "xgboost_direct_missing" else model_name
            )
            tasks.append(
                {
                    "metadata": metadata,
                    "model_name": model_name,
                    "model_params": params,
                    "test_path": test_path,
                    "horizons": horizons,
                    "n_origins": dataset_origins,
                    "seasonal_period": seasonal_period,
                    "metric_keys": metric_keys,
                    "predictions_dir": str(predictions_dir),
                    "seed": int(settings.get("seed", 42)),
                    "include_lead_time_bins": include_lead_time_bins,
                }
            )

    if not tasks:
        raise ValueError("No rolling-origin tasks could be constructed")
    if skipped:
        print(f"Skipped sources without matching test set: {len(skipped)}")

    max_workers = max(1, int(settings.get("max_workers", 1)))
    print("=" * 72)
    print("ROLLING-ORIGIN TEST EVALUATION (AUTHORITATIVE REBUTTAL PATH)")
    print("=" * 72)
    print(f"Models: {selected}")
    print(f"Default origins: {default_origins}; per-dataset overrides: {origin_counts}")
    print(f"Tasks (source x model): {len(tasks)}")
    print(f"Workers: {max_workers}")
    print("Each origin fits once to H_max; cumulative and lead-time metrics are sliced.")
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(config.get_prediction_results_dir()) / (
        f"prediction_results_rolling_origins_{timestamp}.csv"
    )
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_path, index=False)
    aggregate_path = output_root / f"origin_summary_{timestamp}.csv"
    aggregate_origin_results(result_frame, metric_keys).to_csv(aggregate_path, index=False)

    summary_path = output_root / f"run_summary_{timestamp}.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "models": selected,
                "default_origins": default_origins,
                "origin_counts": origin_counts,
                "tasks": len(tasks),
                "successful_metric_rows": len(results),
                "errors": errors,
                "result_file": str(output_path),
                "aggregate_result_file": str(aggregate_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} origin-level metric rows to {output_path}")
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

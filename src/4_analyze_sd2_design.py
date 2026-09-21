#!/usr/bin/env python3
# ruff: noqa: E402
"""Analyze and optionally optimize the local-window SD2 reconstruction design."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from framework.plugin_registry import get_reconstruction_models
from missingness_techniques.mar import apply_mar
from missingness_techniques.mcar import apply_mcar
from missingness_techniques.mnar import apply_mnar
from optimization.sd2_ablation import ENCODINGS, run_ablation_cases
from reconstruction_metrics import compute_metrics_from_series, get_metric_spec
from reconstruction_models.sd2_settings import DEFAULT_PROMPTS
from reconstruction_models.sd2_windowing import plan_reconstruction_windows
from utils.config_loader import load_config

PROMPT_CANDIDATES = {
    "gaf": [
        DEFAULT_PROMPTS["gaf"],
        "grayscale gramian angular field of a continuous sensor signal, preserve mathematical structure",
        "scientific time series gramian angular field, coherent diagonal and smooth local texture",
    ],
    "mtf": [
        DEFAULT_PROMPTS["mtf"],
        "grayscale markov transition field of a sensor signal, preserve transition probabilities",
        "scientific time series markov transition field, coherent state-transition texture",
    ],
    "rp": [
        DEFAULT_PROMPTS["rp"],
        "grayscale continuous recurrence distance plot, symmetric matrix with zero diagonal",
        "scientific sensor recurrence distance matrix, preserve symmetry and local dynamics",
    ],
    "spec": [
        DEFAULT_PROMPTS["spec"],
        "grayscale time frequency spectrogram of a continuous industrial sensor signal",
        "scientific sensor spectrogram, coherent frequency bands and temporal continuity",
    ],
}
MECHANISMS = {"MCAR": apply_mcar, "MAR": apply_mar, "MNAR": apply_mnar}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def load_series(path: Path, train_length: int | None = None) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    series = series.dropna()
    if train_length:
        series = series.iloc[:train_length]
    return series.reset_index(drop=True)


def load_split_manifest(config) -> dict | None:
    manifest_path = Path(config.get_split_manifest_path())
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_hpo_source_dir(config) -> Path:
    """SD2 HPO must read only the sd2_validation partition when available."""
    validation_dir = Path(config.get_splitted_sd2_validation_dir())
    if validation_dir.is_dir() and any(validation_dir.glob("*.csv")):
        return validation_dir
    return Path(config.get_cleaned_dir())


def write_hpo_disjointness_proof(
    output_dir: Path,
    *,
    split_manifest: dict | None,
    hpo_source_dir: Path,
    rolling_test_dir: Path,
) -> Path:
    """Persist evidence that HPO indices do not overlap rolling-test indices."""
    proof = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hpo_source_dir": str(hpo_source_dir),
        "rolling_test_dir": str(rolling_test_dir),
        "datasets": {},
    }
    if split_manifest:
        for dataset, payload in split_manifest.get("datasets", {}).items():
            proof["datasets"][dataset] = {
                "sd2_validation": payload.get("sd2_validation"),
                "rolling_test": payload.get("rolling_test"),
                "disjoint": payload.get("sd2_validation", {}).get("end_exclusive", 0)
                <= payload.get("rolling_test", {}).get("start", 0),
            }
    path = output_dir / "sd2_hpo_disjointness_proof.json"
    path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return path


def expected_marked_axis_fraction(effective_samples: int, image_size: int, missing_rate: float) -> float:
    if effective_samples <= image_size:
        return min(1.0, effective_samples * missing_rate / image_size)
    samples_per_pixel = effective_samples / image_size
    return 1.0 - (1.0 - missing_rate) ** samples_per_pixel


def audit_legacy_dataset(path: Path) -> dict:
    summary_path = path / "dataset_summary.json"
    if not summary_path.exists():
        return {"usable": False, "reason": "dataset_summary.json is missing"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = summary.get("samples", [])
    rates = [float(item["missing_rate"]) for item in samples if "missing_rate" in item]
    lengths = [int(item["length"]) for item in samples if "length" in item]

    def numeric_summary(values: list[float] | list[int]) -> dict[str, float]:
        if not values:
            return {}
        return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}

    resolutions: dict[str, dict[str, int]] = {}
    for encoding in ("gaf", "mtf", "rp", "spec"):
        counts: dict[str, int] = {}
        for image_path in (path / "original").glob(f"*_{encoding}.png"):
            from PIL import Image

            with Image.open(image_path) as image:
                key = f"{image.width}x{image.height}"
            counts[key] = counts.get(key, 0) + 1
        resolutions[encoding] = counts

    return {
        "usable": False,
        "total_series": len(samples),
        "original_images": len(list((path / "original").glob("*.png"))),
        "missing_images": len(list((path / "missing").glob("*.png"))),
        "metadata_files": len(list((path / "masks").glob("*_metadata.json"))),
        "resolutions": resolutions,
        "pattern_distribution": dict(Counter(item.get("pattern_type", "unknown") for item in samples)),
        "missing_type_distribution": dict(Counter(item.get("missing_type", "unknown") for item in samples)),
        "missing_rate": numeric_summary(rates),
        "series_length": numeric_summary(lengths),
        "reasons": [
            "encodings were generated by the superseded GAF/MTF/RP/SPEC code",
            "there are no explicit binary inpainting-mask PNG files",
            "missing images were inferred from image differences after interpolation",
            "image resolutions vary with source length and are later upscaled",
            "the dataset does not use the same local-window contract as inference",
        ],
        "salvage": [
            "pattern and missingness metadata can guide the new generator",
            "the 2,000 synthetic-series distribution can be reproduced from its seed",
            "old images may be retained only as a legacy ablation, not mixed with new training",
        ],
    }


def build_analytical_rows(
    metadata: dict,
    cleaned_dir: Path,
    window_sizes: list[int],
    image_sizes: list[int],
    rates: list[float],
    context_samples: int,
) -> list[dict]:
    rows = []
    for filename, info in metadata["series"].items():
        dataset = Path(filename).stem
        train_length = int(info.get("train_length") or info["n"])
        sampling_seconds = float(info["sampling_interval_seconds"])
        long_horizon = int(info.get("h_long") or 1)

        for window_samples in window_sizes:
            effective = min(window_samples, train_length)
            safe_context = min(context_samples, max(0, (effective - 1) // 2))
            calls = len(plan_reconstruction_windows(train_length, window_samples, safe_context))
            for image_size in image_sizes:
                for rate in rates:
                    axis_mask = expected_marked_axis_fraction(effective, image_size, rate)
                    matrix_mask = 1.0 - (1.0 - axis_mask) ** 2
                    rows.append(
                        {
                            "dataset": dataset,
                            "sampling_seconds": sampling_seconds,
                            "train_length": train_length,
                            "window_samples": window_samples,
                            "effective_window_samples": effective,
                            "window_span": human_duration(effective * sampling_seconds),
                            "image_size": image_size,
                            "samples_per_pixel": effective / image_size,
                            "missing_rate": rate,
                            "axis_mask_fraction": axis_mask,
                            "gaf_mtf_rp_mask_fraction": matrix_mask,
                            "gaf_mtf_rp_context_fraction": 1.0 - matrix_mask,
                            "spec_context_fraction": 1.0 - axis_mask,
                            "windows_per_file": calls,
                            "long_horizon_coverage": min(effective / long_horizon, 1.0),
                            "relative_pixel_memory": (image_size / 512.0) ** 2,
                            "sd2_native_resolution": image_size == 512,
                        }
                    )
    return rows


def structural_recommendations(rows: pd.DataFrame) -> list[dict]:
    recommendations = []
    grouped = rows.groupby(["dataset", "window_samples", "image_size"], sort=True)
    summary = grouped.agg(
        context=("gaf_mtf_rp_context_fraction", "mean"),
        long_horizon_coverage=("long_horizon_coverage", "first"),
        windows_per_file=("windows_per_file", "first"),
        relative_memory=("relative_pixel_memory", "first"),
        native=("sd2_native_resolution", "first"),
        effective_window=("effective_window_samples", "first"),
    ).reset_index()

    for dataset, candidates in summary.groupby("dataset"):
        candidates = candidates.copy()
        max_effective_window = float(candidates["effective_window"].max())
        throughput = candidates["effective_window"] / max_effective_window
        native_factor = np.where(candidates["native"], 1.0, 0.35)
        candidates["structural_score"] = (
            0.55 * candidates["context"] + 0.35 * candidates["long_horizon_coverage"] + 0.10 * throughput
        ) * native_factor
        best = candidates.sort_values(
            ["structural_score", "relative_memory", "window_samples"],
            ascending=[False, True, True],
        ).iloc[0]
        recommendations.append(
            {
                "dataset": dataset,
                "window_samples": int(best["window_samples"]),
                "image_size": int(best["image_size"]),
                "structural_score": float(best["structural_score"]),
                "status": "provisional_analytical",
            }
        )
    return recommendations


def encoding_from_model(model_name: str) -> str:
    for encoding in PROMPT_CANDIDATES:
        if f"_{encoding}" in model_name:
            return encoding
    raise ValueError(f"Unknown encoding in {model_name}")


def validation_slice(source: pd.Series, window_samples: int, case_number: int, n_cases: int) -> pd.Series:
    length = min(window_samples, len(source))
    if length == len(source):
        return source.copy().reset_index(drop=True)
    fraction = (case_number + 1) / (n_cases + 1)
    center = int(fraction * len(source))
    start = min(max(0, center - length // 2), len(source) - length)
    return source.iloc[start : start + length].reset_index(drop=True)


def run_representation_ablations(
    args: argparse.Namespace,
    metadata: dict,
    cleaned_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run CPU-only representation and clean-image oracle controls."""
    rows: list[dict] = []
    for filename, info in metadata["series"].items():
        dataset = Path(filename).stem
        source = load_series(cleaned_dir / filename, int(info.get("train_length") or info["n"]))
        seen_effective_windows: set[int] = set()
        for window_samples in args.window_sizes:
            effective_window = min(window_samples, len(source))
            if effective_window in seen_effective_windows:
                continue
            seen_effective_windows.add(effective_window)
            n_cases = 1 if effective_window == len(source) else args.cases_per_dataset
            for case_number in range(n_cases):
                clean = validation_slice(
                    source,
                    window_samples,
                    case_number,
                    n_cases,
                )
                for image_size in args.image_sizes:
                    with contextlib.redirect_stdout(io.StringIO()):
                        case_rows = run_ablation_cases(
                            clean=clean,
                            encodings=args.ablation_encodings,
                            image_size=image_size,
                            mechanisms=MECHANISMS,
                            rates=args.rates,
                            seed=args.seed + case_number * len(MECHANISMS),
                        )
                    for row in case_rows:
                        row.update(
                            {
                                "dataset": dataset,
                                "window_samples": window_samples,
                                "effective_window_samples": len(clean),
                                "case": case_number,
                            }
                        )
                    rows.extend(case_rows)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "sd2_representation_ablations.csv", index=False)
    return result


def run_empirical_search(
    args: argparse.Namespace,
    metadata: dict,
    hpo_dir: Path,
    output_dir: Path,
    *,
    production_context_samples: int = 64,
) -> tuple[pd.DataFrame, list[dict]]:
    import optuna
    import torch

    if torch.cuda.is_available():
        print("Using CUDA for --run-inference")
    else:
        print(
            "CUDA is not available; --run-inference will run on CPU. "
            "This is supported but much slower than GPU."
        )

    registry = get_reconstruction_models()
    models = args.models or [
        "stable_diffusion_2_gaf",
        "stable_diffusion_2_mtf",
        "stable_diffusion_2_rp",
        "stable_diffusion_2_spec",
    ]
    missing_models = [model for model in models if model not in registry]
    if missing_models:
        raise ValueError(f"Unknown models: {missing_models}")

    metric_spec = get_metric_spec(args.metric)
    trial_rows: list[dict] = []
    winners: list[dict] = []

    for model_name in models:
        encoding = encoding_from_model(model_name)
        for filename, info in metadata["series"].items():
            dataset = Path(filename).stem
            source = load_series(hpo_dir / filename)
            if source.empty:
                raise ValueError(f"Empty SD2 validation series: {hpo_dir / filename}")

            def objective(trial):
                window_samples = trial.suggest_categorical("window_samples", args.window_sizes)
                image_size = trial.suggest_categorical("image_size", args.image_sizes)
                prompt_index = trial.suggest_int("prompt_index", 0, len(PROMPT_CANDIDATES[encoding]) - 1)
                steps = trial.suggest_categorical("num_inference_steps", args.steps)
                guidance = trial.suggest_categorical("guidance_scale", args.guidance)
                prompt = PROMPT_CANDIDATES[encoding][prompt_index]
                losses = []

                for case_number in range(args.cases_per_dataset):
                    clean = validation_slice(
                        source,
                        window_samples,
                        case_number,
                        args.cases_per_dataset,
                    )
                    mechanism_name = list(MECHANISMS)[case_number % len(MECHANISMS)]
                    rate = args.rates[case_number % len(args.rates)]
                    with contextlib.redirect_stdout(io.StringIO()):
                        degraded = MECHANISMS[mechanism_name](clean, rate, seed=args.seed + case_number)
                    started = time.perf_counter()
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            reconstructed = registry[model_name](
                                degraded,
                                seed=args.sd2_seeds[0],
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                window_samples=window_samples,
                                context_samples=production_context_samples,
                                image_size=image_size,
                                prompt=prompt,
                            )
                        metrics = compute_metrics_from_series(clean, degraded, reconstructed)
                        elapsed = time.perf_counter() - started
                        raw = float(metrics[args.metric])
                        loss = raw if metric_spec.lower_is_better else -raw
                        losses.append(loss)
                        trial_rows.append(
                            {
                                "model": model_name,
                                "encoding": encoding,
                                "dataset": dataset,
                                "trial": trial.number,
                                "case": case_number,
                                "mechanism": mechanism_name,
                                "missing_rate": rate,
                                "window_samples": window_samples,
                                "image_size": image_size,
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "num_inference_steps": steps,
                                "guidance_scale": guidance,
                                "sd2_seed": args.sd2_seeds[0],
                                "metric": args.metric,
                                "metric_value": raw,
                                "seconds": elapsed,
                                "status": "success",
                            }
                        )
                    except Exception as exc:
                        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        trial_rows.append(
                            {
                                "model": model_name,
                                "encoding": encoding,
                                "dataset": dataset,
                                "trial": trial.number,
                                "case": case_number,
                                "window_samples": window_samples,
                                "image_size": image_size,
                                "prompt": prompt,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                        return float("inf")
                return float(np.mean(losses)) if losses else float("inf")

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=args.seed),
            )
            study.optimize(objective, n_trials=args.n_trials)
            winner = dict(study.best_params)
            winner.update(
                {
                    "model": model_name,
                    "encoding": encoding,
                    "dataset": dataset,
                    "prompt": PROMPT_CANDIDATES[encoding][winner.pop("prompt_index")],
                    "objective": float(study.best_value),
                    "metric": args.metric,
                    "status": "empirical_gpu",
                }
            )
            winners.append(winner)

    result = pd.DataFrame(trial_rows)
    result.to_csv(output_dir / "sd2_inference_trials.csv", index=False)
    return result, winners


def run_seed_sensitivity(
    args: argparse.Namespace,
    metadata: dict,
    hpo_dir: Path,
    output_dir: Path,
    winners: list[dict],
    *,
    production_context_samples: int = 64,
) -> pd.DataFrame:
    """Repeat winning settings over explicit SD2 seeds on one case per dataset."""
    registry = get_reconstruction_models()
    rows = []
    for winner in winners:
        filename = next(
            name for name in metadata["series"] if Path(name).stem == winner["dataset"]
        )
        info = metadata["series"][filename]
        source = load_series(hpo_dir / filename)
        clean = validation_slice(source, int(winner["window_samples"]), 0, 1)
        mechanism = "MCAR"
        rate = args.rates[len(args.rates) // 2]
        degraded = MECHANISMS[mechanism](clean, rate, seed=args.seed)
        for sd2_seed in args.sd2_seeds:
            started = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    reconstructed = registry[winner["model"]](
                        degraded,
                        seed=sd2_seed,
                        num_inference_steps=int(winner["num_inference_steps"]),
                        guidance_scale=float(winner["guidance_scale"]),
                        window_samples=int(winner["window_samples"]),
                        context_samples=production_context_samples,
                        image_size=int(winner["image_size"]),
                        prompt=winner["prompt"],
                    )
                value = float(
                    compute_metrics_from_series(clean, degraded, reconstructed)[args.metric]
                )
                rows.append(
                    {
                        "model": winner["model"],
                        "encoding": winner["encoding"],
                        "dataset": winner["dataset"],
                        "mechanism": mechanism,
                        "missing_rate": rate,
                        "sd2_seed": sd2_seed,
                        "metric": args.metric,
                        "metric_value": value,
                        "seconds": time.perf_counter() - started,
                        "status": "success",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": winner["model"],
                        "encoding": winner["encoding"],
                        "dataset": winner["dataset"],
                        "sd2_seed": sd2_seed,
                        "status": "error",
                        "error": str(exc),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "sd2_seed_sensitivity.csv", index=False)
    successful = result[result["status"] == "success"]
    if len(successful):
        summary = (
            successful.groupby(["model", "encoding", "dataset", "metric"])["metric_value"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        summary.to_csv(output_dir / "sd2_seed_sensitivity_summary.csv", index=False)
    return result


def write_report(
    output_dir: Path,
    analytical: pd.DataFrame,
    recommendations: list[dict],
    audit: dict,
    search_space: dict,
    empirical: pd.DataFrame | None = None,
    ablations: pd.DataFrame | None = None,
    metric: str = "smape",
) -> None:
    lines = [
        "# SD2 window, resolution, prompt and parameter analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Interpretation",
        "",
        "- window_samples is the number of time-series samples processed locally.",
        "- image_size is the square SD2 input resolution.",
        "- SD2 native resolution is 512x512; larger images are experimental.",
        "- analytical recommendations are provisional until the GPU validation is run.",
        "",
        "## Analytical method",
        "",
        "- Structural score = (0.55 x retained matrix context + 0.35 x long-horizon coverage + 0.10 x relative effective-window size) x native-resolution factor.",
        "- Native-resolution factor is 1.0 at 512 px and a conservative 0.35 at 1024/2048 px because SD2 was trained at 512 px.",
        "- Pixel memory is used as a tie-breaker; reported calls/file are worst-case planned windows.",
        "- This score compares geometry, context and cost. It is not a reconstruction-accuracy measurement.",
        "- The runtime config remains at the conservative 512/512 baseline until GPU validation supplies empirical winners.",
        "",
        "## Recommendations",
        "",
        "| dataset | window samples | image size | status |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in recommendations:
        lines.append(f"| {item['dataset']} | {item['window_samples']} | {item['image_size']} | {item['status']} |")

    lines.extend(
        [
            "",
            "## Existing training dataset audit",
            "",
            f"- Base series: {audit.get('total_series', 0)}",
            f"- Original images: {audit.get('original_images', 0)}",
            f"- Missing images: {audit.get('missing_images', 0)}",
            f"- Directly usable for the corrected pipeline: {audit.get('usable', False)}",
            f"- Pattern distribution: {json.dumps(audit.get('pattern_distribution', {}), sort_keys=True)}",
            f"- Missing-pattern distribution: {json.dumps(audit.get('missing_type_distribution', {}), sort_keys=True)}",
            f"- Missing-rate summary: {json.dumps(audit.get('missing_rate', {}), sort_keys=True)}",
            f"- Series-length summary: {json.dumps(audit.get('series_length', {}), sort_keys=True)}",
            "",
        ]
    )
    for reason in audit.get("reasons", [audit.get("reason", "unknown")]):
        lines.append(f"- Limitation: {reason}")
    for item in audit.get("salvage", []):
        lines.append(f"- Reusable: {item}")

    lines.extend(
        [
            "",
            "## Candidate summary",
            "",
            "| dataset | window | span | image | samples/pixel | mean retained matrix context | potential calls/file | memory vs 512 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    summary = (
        analytical.groupby(["dataset", "window_samples", "image_size"])
        .agg(
            context=("gaf_mtf_rp_context_fraction", "mean"),
            calls=("windows_per_file", "first"),
            span=("window_span", "first"),
            samples_per_pixel=("samples_per_pixel", "first"),
            memory=("relative_pixel_memory", "first"),
        )
        .reset_index()
    )
    for row in summary.itertuples():
        lines.append(
            f"| {row.dataset} | {row.window_samples} | {row.span} | {row.image_size} | "
            f"{row.samples_per_pixel:.2f} | {row.context:.1%} | {row.calls} | {row.memory:.1f}x |"
        )

    lines.extend(
        [
            "",
            "## GPU optimization search space",
            "",
            f"- Window samples: {search_space['window_samples']}",
            f"- Image sizes: {search_space['image_size']}",
            f"- Inference steps: {search_space['num_inference_steps']}",
            f"- Guidance scales: {search_space['guidance_scale']}",
            f"- Missing rates: {search_space['missing_rates']}",
            f"- Explicit SD2 sensitivity seeds: {search_space['sd2_seeds']}",
            f"- Trials per model/dataset: {search_space['trials_per_model_dataset']}",
            f"- Validation cases per trial: {search_space['cases_per_trial']}",
            "",
            "Representation-specific prompt candidates:",
        ]
    )
    for encoding, prompts in search_space["prompts"].items():
        lines.append(f"- {encoding.upper()}: " + " | ".join(prompts))

    if empirical is not None:
        successful = int((empirical.get("status") == "success").sum())
        lines.extend(
            [
                "",
                "## GPU validation",
                "",
                f"Successful validation cases: {successful}",
                "Detailed results: sd2_inference_trials.csv",
            ]
        )

    if ablations is not None and not ablations.empty:
        lines.extend(
            [
                "",
                "## Round-trip and oracle ablations",
                "",
                "- `round_trip` scores all clean samples after encode, 8-bit image quantization, and decode; diffusion is bypassed.",
                "- `oracle_clean_image` scores only simulated missing positions after substituting the ideal clean encoded image for the SD2 output.",
                "- The oracle uses clean-image decoder metadata and is a representation ceiling, not a deployable imputation method.",
                f"- Summary metric: {metric}. Detailed results: sd2_representation_ablations.csv",
                "",
                "| ablation | encoding | image | mean metric | cases |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        summary = (
            ablations.groupby(["ablation", "encoding", "image_size"], sort=True)[metric]
            .agg(["mean", "count"])
            .reset_index()
        )
        for row in summary.itertuples():
            lines.append(
                f"| {row.ablation} | {row.encoding.upper()} | {row.image_size} | "
                f"{row.mean:.6g} | {row.count} |"
            )

    (output_dir / "sd2_design_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", default="data/1_6_sd2_optimization")
    parser.add_argument("--window-sizes", default="512,1024,2048")
    parser.add_argument("--image-sizes", default="512,1024,2048")
    parser.add_argument("--steps", default="20,30,42,50")
    parser.add_argument("--guidance", default="1.0,3.0,5.0,7.5")
    parser.add_argument("--rates", default="0.03,0.08,0.20")
    parser.add_argument("--context-samples", type=int, default=64)
    parser.add_argument("--metric", default="smape")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--cases-per-dataset", type=int, default=3)
    parser.add_argument("--models", nargs="*")
    parser.add_argument(
        "--ablation-encodings",
        nargs="+",
        choices=ENCODINGS,
        default=list(ENCODINGS),
        help="Representations evaluated by --run-ablation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sd2-seeds",
        default="42,43,44,45,46",
        help="Five explicit Diffusers seeds used for winner sensitivity analysis.",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run CPU-only round-trip and clean-image oracle ablations.",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run Optuna search with actual SD2 inpainting (GPU preferred; CPU is allowed but slow).",
    )
    args = parser.parse_args()

    args.window_sizes = parse_int_list(args.window_sizes)
    args.image_sizes = parse_int_list(args.image_sizes)
    args.steps = parse_int_list(args.steps)
    args.guidance = parse_float_list(args.guidance)
    args.rates = parse_float_list(args.rates)
    args.sd2_seeds = parse_int_list(args.sd2_seeds)
    if not 3 <= len(args.sd2_seeds) <= 5:
        raise ValueError("--sd2-seeds must contain between 3 and 5 seeds")

    config = load_config(args.config)
    metadata_path = Path(config.config["split"]["horizons"]["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cleaned_dir = Path(config.get_cleaned_dir())
    hpo_dir = resolve_hpo_source_dir(config)
    split_manifest = load_split_manifest(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    production_context = int(
        config.config.get("computation", {})
        .get("stable_diffusion", {})
        .get("windowing", {})
        .get("context_samples", args.context_samples)
    )

    rows = build_analytical_rows(
        metadata,
        cleaned_dir,
        args.window_sizes,
        args.image_sizes,
        args.rates,
        args.context_samples,
    )
    analytical = pd.DataFrame(rows)
    analytical.to_csv(output_dir / "sd2_design_analysis.csv", index=False)
    recommendations = structural_recommendations(analytical)
    audit = audit_legacy_dataset(Path("stdiff_training_data"))

    empirical = None
    ablations = None
    if args.run_ablation:
        ablations = run_representation_ablations(args, metadata, cleaned_dir, output_dir)

    if args.run_inference:
        if hpo_dir == cleaned_dir:
            print(
                "⚠️  SD2 validation partition not found; HPO falls back to cleaned data. "
                "Run make create-split before GPU HPO for leakage-free selection."
            )
        write_hpo_disjointness_proof(
            output_dir,
            split_manifest=split_manifest,
            hpo_source_dir=hpo_dir,
            rolling_test_dir=Path(config.get_splitted_test_dir()),
        )
        empirical, empirical_winners = run_empirical_search(
            args,
            metadata,
            hpo_dir,
            output_dir,
            production_context_samples=production_context,
        )
        run_seed_sensitivity(
            args,
            metadata,
            hpo_dir,
            output_dir,
            empirical_winners,
            production_context_samples=production_context,
        )
        recommendations = empirical_winners
        runtime_path = output_dir / "sd2_runtime_overrides.json"
        if runtime_path.is_file():
            previous_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            overrides: dict[str, dict[str, dict]] = previous_runtime.get("overrides", {})
        else:
            overrides = {}

        runtime_keys = (
            "window_samples",
            "image_size",
            "num_inference_steps",
            "guidance_scale",
            "prompt",
        )
        for winner in empirical_winners:
            overrides.setdefault(winner["dataset"], {})[winner["model"]] = {
                key: winner[key] for key in runtime_keys
            }
        runtime_payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "empirical GPU optimization",
            "overrides": overrides,
        }
        runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")

    search_space = {
        "window_samples": args.window_sizes,
        "image_size": args.image_sizes,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "missing_rates": args.rates,
        "sd2_seeds": args.sd2_seeds,
        "prompts": PROMPT_CANDIDATES,
        "trials_per_model_dataset": args.n_trials,
        "cases_per_trial": args.cases_per_dataset,
    }

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recommendations": recommendations,
        "search_space": search_space,
        "legacy_dataset_audit": audit,
        "ablation_output": (
            "sd2_representation_ablations.csv" if args.run_ablation else None
        ),
        "note": "Analytical winners are provisional; use --run-inference for model-based selection.",
    }
    (output_dir / "sd2_recommendations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        output_dir,
        analytical,
        recommendations,
        audit,
        search_space,
        empirical,
        ablations,
        args.metric,
    )
    print(f"Analysis written to {output_dir}")


if __name__ == "__main__":
    main()

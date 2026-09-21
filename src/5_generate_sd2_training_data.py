#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate window-aligned training triplets for SD2 time-series inpainting."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

from missingness_techniques.structured import apply_structured_missingness
from utils.experiment_naming import MISSINGNESS_STRUCTURES
from utils.missingness_analysis import summarize_missingness
from reconstruction_models.sd2_settings import DEFAULT_PROMPTS
from reconstruction_models.stable_diffusion_2_gaf import series_to_gaf
from reconstruction_models.stable_diffusion_2_mtf import series_to_mtf
from reconstruction_models.stable_diffusion_2_rp import series_to_rp
from reconstruction_models.stable_diffusion_2_spec import series_to_spectrogram

ENCODINGS = ("gaf", "mtf", "rp", "spec")
MECHANISMS = ("MCAR", "MAR", "MNAR")


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


class SyntheticSeriesGenerator:
    """Generate reproducible signals broader than the legacy seven templates."""

    PATTERNS = (
        "harmonic",
        "trend",
        "seasonal",
        "autoregressive",
        "random_walk",
        "steps",
        "spikes",
        "chirp",
        "heteroskedastic",
        "mixed",
    )

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate(self, length: int, pattern: str) -> np.ndarray:
        values, _ = self.generate_with_metadata(length, pattern)
        return values

    def generate_with_metadata(self, length: int, pattern: str) -> tuple[np.ndarray, dict]:
        t = np.linspace(0.0, 1.0, length)
        r = self.rng
        params: dict = {"pattern": pattern, "length": length}

        if pattern == "harmonic":
            params.update(
                {
                    "amplitude": float(r.uniform(0.5, 2.0)),
                    "frequency": float(r.uniform(1, 12)),
                    "phase": float(r.uniform(0, 2 * np.pi)),
                }
            )
            values = params["amplitude"] * np.sin(
                2 * np.pi * params["frequency"] * t + params["phase"]
            )
        elif pattern == "trend":
            params.update(
                {
                    "linear_coef": float(r.uniform(-3, 3)),
                    "quadratic_coef": float(r.uniform(-1, 1)),
                }
            )
            values = params["linear_coef"] * t + params["quadratic_coef"] * t**2
        elif pattern == "seasonal":
            n_components = int(r.integers(2, 6))
            components = []
            for _ in range(n_components):
                components.append(
                    {
                        "amplitude": float(r.uniform(0.2, 1.2)),
                        "frequency": float(r.uniform(1, 20)),
                        "phase": float(r.uniform(0, 2 * np.pi)),
                    }
                )
            params["components"] = components
            values = sum(
                comp["amplitude"]
                * np.sin(2 * np.pi * comp["frequency"] * t + comp["phase"])
                for comp in components
            )
        elif pattern == "autoregressive":
            params.update(
                {
                    "phi": float(r.uniform(0.3, 0.98)),
                    "noise_scale": float(r.uniform(0.05, 0.4)),
                }
            )
            noise = r.normal(0, params["noise_scale"], length)
            values = np.zeros(length)
            for idx in range(1, length):
                values[idx] = params["phi"] * values[idx - 1] + noise[idx]
        elif pattern == "random_walk":
            params["step_scale"] = float(r.uniform(0.02, 0.2))
            values = np.cumsum(r.normal(0, params["step_scale"], length))
        elif pattern == "steps":
            n_cuts = int(r.integers(2, 8))
            cuts = sorted(r.choice(np.arange(1, length - 1), n_cuts, replace=False))
            params["cuts"] = [int(value) for value in cuts]
            values = np.zeros(length)
            levels = []
            for left, right in zip([0, *cuts], [*cuts, length]):
                level = float(r.uniform(-2, 2))
                levels.append(level)
                values[left:right] = level
            params["levels"] = levels
        elif pattern == "spikes":
            n_spikes = max(1, length // 40)
            indices = r.choice(length, n_spikes, replace=False)
            params["spike_indices"] = [int(value) for value in indices]
            params["baseline_scale"] = 0.05
            values = r.normal(0, params["baseline_scale"], length)
            values[indices] += r.normal(0, 2.0, len(indices))
        elif pattern == "chirp":
            params.update(
                {
                    "start_frequency": float(r.uniform(1, 4)),
                    "stop_frequency": float(r.uniform(8, 30)),
                }
            )
            phase = 2 * np.pi * (
                params["start_frequency"] * t
                + 0.5 * (params["stop_frequency"] - params["start_frequency"]) * t**2
            )
            values = np.sin(phase)
        elif pattern == "heteroskedastic":
            params.update(
                {
                    "start_scale": float(r.uniform(0.02, 0.2)),
                    "stop_scale": float(r.uniform(0.3, 1.0)),
                }
            )
            scale = np.linspace(params["start_scale"], params["stop_scale"], length)
            values = r.normal(0, scale)
        elif pattern == "mixed":
            seasonal, seasonal_params = self.generate_with_metadata(length, "seasonal")
            trend, trend_params = self.generate_with_metadata(length, "trend")
            ar, ar_params = self.generate_with_metadata(length, "autoregressive")
            params["mixed_components"] = {
                "seasonal": seasonal_params,
                "trend": trend_params,
                "autoregressive": ar_params,
            }
            values = seasonal + 0.5 * trend + 0.3 * ar
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        params["observation_noise_scale"] = float(r.uniform(0.002, 0.08))
        values = values + r.normal(0, params["observation_noise_scale"], length)
        if not np.isfinite(values).all():
            raise ValueError("Synthetic generator produced non-finite values")
        return values.astype(np.float64), params


def degrade(
    clean: pd.Series,
    mechanism: str,
    structure: str,
    rate: float,
    seed: int,
) -> pd.Series:
    return apply_structured_missingness(
        clean,
        rate,
        mechanism,
        structure,
        seed=seed,
    )


def sample_to_pixel(sample: int, length: int, image_size: int) -> int:
    if length <= 1:
        return 0
    return int(round(sample * (image_size - 1) / (length - 1)))


def image_mask(
    missing: np.ndarray,
    encoding: str,
    image_size: int,
) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for sample in np.flatnonzero(missing):
        pixel = sample_to_pixel(int(sample), len(missing), image_size)
        if encoding in {"gaf", "mtf", "rp"}:
            mask[pixel, :] = 255
            mask[:, pixel] = 255
        else:
            mask[:, pixel] = 255
    return mask


def encode_pair(
    clean: pd.Series,
    conditioning: pd.Series,
    encoding: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if encoding == "gaf":
        target = series_to_gaf(clean, size=image_size)
        source = series_to_gaf(conditioning, size=image_size)
        return (
            np.clip((target + 1.0) * 127.5, 0, 255).astype(np.uint8),
            np.clip((source + 1.0) * 127.5, 0, 255).astype(np.uint8),
        )
    if encoding == "mtf":
        target = series_to_mtf(clean, size=image_size)
        source = series_to_mtf(conditioning, size=image_size)
        return (
            np.clip(target * 255.0, 0, 255).astype(np.uint8),
            np.clip(source * 255.0, 0, 255).astype(np.uint8),
        )
    if encoding == "rp":
        target = series_to_rp(clean, size=image_size)
        source = series_to_rp(conditioning, size=image_size)
        return (
            np.clip(target * 255.0, 0, 255).astype(np.uint8),
            np.clip(source * 255.0, 0, 255).astype(np.uint8),
        )
    if encoding == "spec":
        target = series_to_spectrogram(clean, size=image_size)
        source = series_to_spectrogram(conditioning, size=image_size)
        lower, upper = float(source.min()), float(source.max())
        scale = upper - lower
        if scale <= 0:
            return (
                np.zeros_like(target, dtype=np.uint8),
                np.zeros_like(source, dtype=np.uint8),
            )
        return (
            np.clip((target - lower) / scale * 255.0, 0, 255).astype(np.uint8),
            np.clip((source - lower) / scale * 255.0, 0, 255).astype(np.uint8),
        )
    raise ValueError(encoding)


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="L").convert("RGB").save(path)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty; use --overwrite or choose another output")
        shutil.rmtree(path)
    for directory in ("clean", "conditioning", "masks", "series"):
        (path / directory).mkdir(parents=True, exist_ok=True)


def load_real_series(cleaned_dir: Path) -> list[tuple[str, pd.Series]]:
    result = []
    for path in sorted(cleaned_dir.glob("*.csv")):
        frame = pd.read_csv(path, index_col=0)
        series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
        if len(series) >= 2:
            result.append((path.stem, series.reset_index(drop=True)))
    return result


def choose_real_window(
    series: pd.Series,
    length: int,
    rng: np.random.Generator,
) -> pd.Series:
    if len(series) <= length:
        x_old = np.linspace(0.0, 1.0, len(series))
        x_new = np.linspace(0.0, 1.0, length)
        return pd.Series(np.interp(x_new, x_old, series.to_numpy(dtype=float)))
    start = int(rng.integers(0, len(series) - length + 1))
    return series.iloc[start : start + length].reset_index(drop=True)



def generate_training_sample(
    series_id: int,
    *,
    output: Path,
    image_size: int,
    window_sizes: list[int],
    rates: list[float],
    mechanisms: list[str],
    structures: list[str],
    source: str,
    real_share: float,
    real_series: list[tuple[str, pd.Series]],
    seed: int,
) -> tuple[list[dict], str]:
    """Generate one independent base series and its four encoded triplets."""
    rng = np.random.default_rng(seed + series_id)
    synthetic = SyntheticSeriesGenerator(rng)
    window_samples = int(rng.choice(window_sizes))
    use_real = source == "mixed" and bool(real_series) and rng.random() < real_share
    generator_params: dict = {}
    if use_real:
        source_name, source_series = real_series[int(rng.integers(len(real_series)))]
        clean = choose_real_window(source_series, window_samples, rng)
        pattern = "real"
        source_kind = "real"
    else:
        pattern = str(rng.choice(SyntheticSeriesGenerator.PATTERNS))
        generated, generator_params = synthetic.generate_with_metadata(window_samples, pattern)
        clean = pd.Series(generated)
        source_name = pattern
        source_kind = "synthetic"

    series_path = output / "series" / f"{series_id:06d}.npy"
    np.save(series_path, clean.to_numpy(dtype=np.float64))
    mechanism = str(rng.choice(mechanisms))
    structure = str(rng.choice(structures))
    rate = float(rng.choice(rates))
    pair_seed = seed + series_id
    corrupted = degrade(clean, mechanism, structure, rate, pair_seed)
    gap_summary, _ = summarize_missingness(corrupted)
    filled = corrupted.interpolate(method="linear", limit_direction="both")
    filled = filled.fillna(float(clean.mean()))
    missing = corrupted.isna().to_numpy()

    records = []
    for encoding in ENCODINGS:
        target, condition = encode_pair(clean, filled, encoding, image_size)
        mask = image_mask(missing, encoding, image_size)
        stem = f"{series_id:06d}_{encoding}"
        target_path = output / "clean" / f"{stem}.png"
        condition_path = output / "conditioning" / f"{stem}.png"
        mask_path = output / "masks" / f"{stem}.png"
        save_rgb(target, target_path)
        save_rgb(condition, condition_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        records.append({
            "series_id": series_id, "encoding": encoding, "source_kind": source_kind,
            "source_name": source_name, "pattern": pattern,
            "generator_params": generator_params if source_kind == "synthetic" else {"source_name": source_name},
            "generator_version": "SyntheticSeriesGenerator.v2",
            "series_path": str(series_path.relative_to(output)),
            "window_samples": window_samples, "image_size": image_size,
            "mechanism": mechanism, "structure": structure,
            "missing_rate_requested": rate, "missing_rate_actual": float(missing.mean()),
            "n_gaps": gap_summary["n_gaps"],
            "gap_length_samples_mean": gap_summary["gap_length_samples_mean"],
            "gap_length_samples_median": gap_summary["gap_length_samples_median"],
            "gap_length_samples_p90": gap_summary["gap_length_samples_p90"],
            "gap_length_samples_max": gap_summary["gap_length_samples_max"],
            "singleton_gap_percent": gap_summary["singleton_gap_percent"],
            "seed": pair_seed, "prompt": DEFAULT_PROMPTS[encoding],
            "clean": str(target_path.relative_to(output)),
            "conditioning": str(condition_path.relative_to(output)),
            "mask": str(mask_path.relative_to(output)),
        })
    return records, source_kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", default="data/sd2_windowed_training")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--window-sizes", default="512,1024,2048")
    parser.add_argument("--rates", default="0.03,0.08,0.20")
    parser.add_argument("--mechanisms", default="MCAR,MAR,MNAR")
    parser.add_argument("--structures", default="scattered,contiguous,mixed")
    parser.add_argument("--source", choices=["synthetic", "mixed"], default="synthetic")
    parser.add_argument("--real-share", type=float, default=0.25)
    parser.add_argument("--cleaned-dir", default="data/1_cleaned_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be positive")

    if args.image_size < 64 or args.image_size % 8:
        raise ValueError("--image-size must be >=64 and divisible by 8")
    if not 0.0 <= args.real_share <= 1.0:
        raise ValueError("--real-share must be in [0, 1]")

    window_sizes = parse_int_list(args.window_sizes)
    rates = parse_float_list(args.rates)
    mechanisms = [item.strip().upper() for item in args.mechanisms.split(",") if item.strip()]
    structures = [item.strip().lower() for item in args.structures.split(",") if item.strip()]
    if not window_sizes or any(size < 16 for size in window_sizes):
        raise ValueError("--window-sizes must contain values >=16")
    if not rates or any(rate <= 0.0 or rate >= 1.0 for rate in rates):
        raise ValueError("--rates must contain values in (0, 1)")
    if not mechanisms:
        raise ValueError("--mechanisms must not be empty")
    unknown = set(mechanisms) - set(MECHANISMS)
    if unknown:
        raise ValueError(f"Unknown mechanisms: {sorted(unknown)}")
    unknown_structures = set(structures) - set(MISSINGNESS_STRUCTURES)
    if not structures or unknown_structures:
        raise ValueError(f"Unknown structures: {sorted(unknown_structures)}")

    output = Path(args.output)
    rng = np.random.default_rng(args.seed)
    synthetic = SyntheticSeriesGenerator(rng)
    real_series = load_real_series(Path(args.cleaned_dir)) if args.source == "mixed" else []

    if args.source == "mixed" and args.real_share > 0 and not real_series:
        raise ValueError(f"No real CSV series found in {args.cleaned_dir}")
    prepare_output(output, args.overwrite)

    manifest_path = output / "manifest.jsonl"
    generated = Parallel(n_jobs=args.workers, backend="loky")(
        delayed(generate_training_sample)(
            series_id, output=output, image_size=args.image_size,
            window_sizes=window_sizes, rates=rates, mechanisms=mechanisms,
            structures=structures, source=args.source, real_share=args.real_share,
            real_series=real_series, seed=args.seed,
        )
        for series_id in tqdm(range(args.samples), desc="Scheduling SD2 triplets")
    )
    counts = {"synthetic": 0, "real": 0}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for records, source_kind in generated:
            counts[source_kind] += 1
            for record in records:
                manifest.write(json.dumps(record) + "\n")

    summary = {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_series": args.samples,
        "training_triplets": args.samples * len(ENCODINGS),
        "image_size": args.image_size,
        "window_sizes": window_sizes,
        "rates": rates,
        "mechanisms": mechanisms,
        "structures": structures,
        "encodings": list(ENCODINGS),
        "source_counts": counts,
        "contract": {
            "clean": "target image",
            "conditioning": "linearly filled corrupted series encoded as image",
            "mask": "white pixels are inpainted; black pixels are preserved",
        },
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated {summary['training_triplets']} triplets in {output}")


if __name__ == "__main__":
    main()

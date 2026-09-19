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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from missingness_techniques.mar import apply_mar
from missingness_techniques.mcar import apply_mcar
from missingness_techniques.mnar import apply_mnar
from reconstruction_models.sd2_settings import DEFAULT_PROMPTS
from reconstruction_models.stable_diffusion_2_gaf import series_to_gaf
from reconstruction_models.stable_diffusion_2_mtf import series_to_mtf
from reconstruction_models.stable_diffusion_2_rp import series_to_rp
from reconstruction_models.stable_diffusion_2_spec import series_to_spectrogram

ENCODINGS = ("gaf", "mtf", "rp", "spec")
MECHANISMS = {"MCAR": apply_mcar, "MAR": apply_mar, "MNAR": apply_mnar}


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
        t = np.linspace(0.0, 1.0, length)
        r = self.rng

        if pattern == "harmonic":
            values = r.uniform(0.5, 2.0) * np.sin(2 * np.pi * r.uniform(1, 12) * t + r.uniform(0, 2 * np.pi))
        elif pattern == "trend":
            values = r.uniform(-3, 3) * t + r.uniform(-1, 1) * t**2
        elif pattern == "seasonal":
            values = sum(
                r.uniform(0.2, 1.2) * np.sin(2 * np.pi * r.uniform(1, 20) * t + r.uniform(0, 2 * np.pi))
                for _ in range(r.integers(2, 6))
            )
        elif pattern == "autoregressive":
            phi = r.uniform(0.3, 0.98)
            noise = r.normal(0, r.uniform(0.05, 0.4), length)
            values = np.zeros(length)
            for idx in range(1, length):
                values[idx] = phi * values[idx - 1] + noise[idx]
        elif pattern == "random_walk":
            values = np.cumsum(r.normal(0, r.uniform(0.02, 0.2), length))
        elif pattern == "steps":
            values = np.zeros(length)
            cuts = sorted(r.choice(np.arange(1, length - 1), r.integers(2, 8), replace=False))
            for left, right in zip([0, *cuts], [*cuts, length]):
                values[left:right] = r.uniform(-2, 2)
        elif pattern == "spikes":
            values = r.normal(0, 0.05, length)
            indices = r.choice(length, max(1, length // 40), replace=False)
            values[indices] += r.normal(0, 2.0, len(indices))
        elif pattern == "chirp":
            start, stop = r.uniform(1, 4), r.uniform(8, 30)
            phase = 2 * np.pi * (start * t + 0.5 * (stop - start) * t**2)
            values = np.sin(phase)
        elif pattern == "heteroskedastic":
            scale = np.linspace(r.uniform(0.02, 0.2), r.uniform(0.3, 1.0), length)
            values = r.normal(0, scale)
        elif pattern == "mixed":
            values = (
                self.generate(length, "seasonal")
                + 0.5 * self.generate(length, "trend")
                + 0.3 * self.generate(length, "autoregressive")
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        values = values + r.normal(0, r.uniform(0.002, 0.08), length)
        if not np.isfinite(values).all():
            raise ValueError("Synthetic generator produced non-finite values")
        return values.astype(np.float64)


def structural_missing(
    series: pd.Series,
    kind: str,
    rate: float,
    rng: np.random.Generator,
) -> pd.Series:
    result = series.copy()
    count = max(1, int(round(len(series) * rate)))
    if kind == "BLOCK":
        blocks = int(rng.integers(1, min(5, count) + 1))
        sizes = rng.multinomial(count, np.full(blocks, 1.0 / blocks))
        selected: set[int] = set()
        for size in sizes:
            if size:
                start = int(rng.integers(0, max(1, len(series) - size + 1)))
                selected.update(range(start, min(start + size, len(series))))
        result.iloc[sorted(selected)] = np.nan
    elif kind == "PERIODIC":
        period = max(2, int(round(1.0 / rate)))
        offset = int(rng.integers(0, period))
        result.iloc[offset::period] = np.nan
    elif kind == "EDGE":
        if rng.random() < 0.5:
            result.iloc[:count] = np.nan
        else:
            result.iloc[-count:] = np.nan
    else:
        raise ValueError(kind)
    return result


def degrade(
    clean: pd.Series,
    mechanism: str,
    rate: float,
    seed: int,
) -> pd.Series:
    if mechanism in MECHANISMS:
        with contextlib.redirect_stdout(io.StringIO()):
            return MECHANISMS[mechanism](clean, rate, seed=seed)
    return structural_missing(clean, mechanism, rate, np.random.default_rng(seed))


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
    for directory in ("clean", "conditioning", "masks"):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", default="data/sd2_windowed_training")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--window-sizes", default="512,1024,2048")
    parser.add_argument("--rates", default="0.03,0.08,0.20")
    parser.add_argument("--mechanisms", default="MCAR,MAR,MNAR,BLOCK,PERIODIC,EDGE")
    parser.add_argument("--source", choices=["synthetic", "mixed"], default="synthetic")
    parser.add_argument("--real-share", type=float, default=0.25)
    parser.add_argument("--cleaned-dir", default="data/1_cleaned_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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
    if not window_sizes or any(size < 16 for size in window_sizes):
        raise ValueError("--window-sizes must contain values >=16")
    if not rates or any(rate <= 0.0 or rate >= 1.0 for rate in rates):
        raise ValueError("--rates must contain values in (0, 1)")
    if not mechanisms:
        raise ValueError("--mechanisms must not be empty")
    unknown = set(mechanisms) - (set(MECHANISMS) | {"BLOCK", "PERIODIC", "EDGE"})
    if unknown:
        raise ValueError(f"Unknown mechanisms: {sorted(unknown)}")

    output = Path(args.output)
    rng = np.random.default_rng(args.seed)
    synthetic = SyntheticSeriesGenerator(rng)
    real_series = load_real_series(Path(args.cleaned_dir)) if args.source == "mixed" else []

    if args.source == "mixed" and args.real_share > 0 and not real_series:
        raise ValueError(f"No real CSV series found in {args.cleaned_dir}")
    prepare_output(output, args.overwrite)

    manifest_path = output / "manifest.jsonl"
    counts = {"synthetic": 0, "real": 0}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for series_id in tqdm(range(args.samples), desc="Generating SD2 triplets"):
            window_samples = int(rng.choice(window_sizes))
            use_real = bool(real_series) and rng.random() < args.real_share
            if use_real:
                source_name, source_series = real_series[int(rng.integers(len(real_series)))]
                clean = choose_real_window(source_series, window_samples, rng)
                pattern = "real"
                source_kind = "real"
            else:
                pattern = str(rng.choice(SyntheticSeriesGenerator.PATTERNS))
                clean = pd.Series(synthetic.generate(window_samples, pattern))
                source_name = pattern
                source_kind = "synthetic"
            counts[source_kind] += 1

            mechanism = str(rng.choice(mechanisms))
            rate = float(rng.choice(rates))
            pair_seed = args.seed + series_id
            corrupted = degrade(clean, mechanism, rate, pair_seed)
            filled = corrupted.interpolate(method="linear", limit_direction="both")
            filled = filled.fillna(float(clean.mean()))
            missing = corrupted.isna().to_numpy()

            for encoding in ENCODINGS:
                target, condition = encode_pair(clean, filled, encoding, args.image_size)
                mask = image_mask(missing, encoding, args.image_size)
                stem = f"{series_id:06d}_{encoding}"
                target_path = output / "clean" / f"{stem}.png"
                condition_path = output / "conditioning" / f"{stem}.png"
                mask_path = output / "masks" / f"{stem}.png"
                save_rgb(target, target_path)
                save_rgb(condition, condition_path)
                Image.fromarray(mask, mode="L").save(mask_path)

                record = {
                    "series_id": series_id,
                    "encoding": encoding,
                    "source_kind": source_kind,
                    "source_name": source_name,
                    "pattern": pattern,
                    "window_samples": window_samples,
                    "image_size": args.image_size,
                    "mechanism": mechanism,
                    "missing_rate_requested": rate,
                    "missing_rate_actual": float(missing.mean()),
                    "seed": pair_seed,
                    "prompt": DEFAULT_PROMPTS[encoding],
                    "clean": str(target_path.relative_to(output)),
                    "conditioning": str(condition_path.relative_to(output)),
                    "mask": str(mask_path.relative_to(output)),
                }
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

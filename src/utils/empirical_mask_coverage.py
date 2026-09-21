"""Empirical SD2 mask coverage from generated degradation realizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from utils.progress import tqdm

from reconstruction_models.sd2_windowing import plan_reconstruction_windows
from utils.experiment_naming import decode_missingness_label


def _parse_degraded_filename(path: Path) -> dict[str, Any] | None:
    stem = path.stem
    parts = stem.split("_")
    rate_idx = next(
        (index for index, part in enumerate(parts) if part.endswith("p") and part[:-1].isdigit()),
        None,
    )
    if rate_idx is None or rate_idx < 1 or rate_idx + 1 >= len(parts):
        return None
    technique, structure = decode_missingness_label(parts[rate_idx - 1])
    return {
        "dataset_name": "_".join(parts[: rate_idx - 1]),
        "technique": technique,
        "structure": structure,
        "rate_percent": int(parts[rate_idx][:-1]),
        "iteration": int(parts[rate_idx + 1]),
    }


def empirical_mask_fraction(
    missing: np.ndarray,
    encoding: str,
    image_size: int,
) -> float:
    """Fraction of inpainting mask pixels marked for reconstruction."""
    if len(missing) == 0:
        return 0.0
    mask = np.zeros((image_size, image_size), dtype=bool)
    for sample in np.flatnonzero(missing):
        pixel = int(round(sample * (image_size - 1) / max(len(missing) - 1, 1)))
        if encoding in {"gaf", "mtf", "rp"}:
            mask[pixel, :] = True
            mask[:, pixel] = True
        else:
            mask[:, pixel] = True
    return float(mask.mean())


def report_empirical_mask_coverage(
    missing_dir: str | Path,
    *,
    representations: tuple[str, ...] = ("gaf", "mtf", "rp", "spec"),
    image_size: int = 512,
    window_samples: int = 512,
    context_samples: int = 64,
) -> pd.DataFrame:
    """Measure mask coverage per dataset × mechanism × structure × rate × iteration × representation."""
    rows: list[dict[str, Any]] = []
    missing_dir = Path(missing_dir)
    files = sorted(missing_dir.glob("*.csv"))
    for path in tqdm(
        files,
        desc="Empirical mask coverage",
        unit="file",
        dynamic_ncols=True,
    ):
        meta = _parse_degraded_filename(path)
        if meta is None:
            continue
        frame = pd.read_csv(path, index_col=0)
        series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        missing = series.isna().to_numpy()
        train_length = len(series)
        windows = plan_reconstruction_windows(train_length, window_samples, context_samples)
        for encoding in representations:
            whole_series_fraction = empirical_mask_fraction(missing, encoding, image_size)
            window_fractions = []
            for window in windows:
                local_missing = missing[window.core_start : window.core_stop]
                window_fractions.append(
                    empirical_mask_fraction(local_missing, encoding, image_size)
                )
            rows.append(
                {
                    **meta,
                    "representation": encoding,
                    "image_size": image_size,
                    "window_samples": window_samples,
                    "context_samples": context_samples,
                    "train_length": train_length,
                    "n_windows": len(windows),
                    "empirical_mask_fraction_whole_series": whole_series_fraction,
                    "empirical_mask_fraction_window_mean": float(np.mean(window_fractions))
                    if window_fractions
                    else np.nan,
                    "empirical_mask_fraction_window_max": float(np.max(window_fractions))
                    if window_fractions
                    else np.nan,
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)

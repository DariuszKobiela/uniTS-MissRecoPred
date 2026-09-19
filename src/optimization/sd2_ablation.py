"""Round-trip and clean-image oracle ablations for SD2 representations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from reconstruction_models.stable_diffusion_2_gaf import gaf_to_series, series_to_gaf
from reconstruction_models.stable_diffusion_2_mtf import mtf_to_series, series_to_mtf
from reconstruction_models.stable_diffusion_2_rp import rp_to_series, series_to_rp
from reconstruction_models.stable_diffusion_2_spec import (
    series_to_spectrogram,
    spectrogram_to_series,
)

ENCODINGS = ("gaf", "mtf", "rp", "spec")


def _uint8_round_trip(
    image: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Apply the same 8-bit grayscale quantization used by SD2 adapters."""
    if upper <= lower:
        return np.full_like(image, lower, dtype=np.float64)
    pixels = np.clip((image - lower) / (upper - lower) * 255.0, 0.0, 255.0).astype(np.uint8)
    return pixels.astype(np.float64) / 255.0 * (upper - lower) + lower


def round_trip_series(
    clean: pd.Series,
    encoding: str,
    image_size: int,
) -> pd.Series:
    """Encode, quantize, and decode a clean series without diffusion."""
    if encoding == "gaf":
        image = series_to_gaf(clean, size=image_size)
        decoded = gaf_to_series(
            _uint8_round_trip(image, -1.0, 1.0),
            len(clean),
            clean,
        )
    elif encoding == "mtf":
        image, metadata = series_to_mtf(clean, size=image_size, return_metadata=True)
        decoded = mtf_to_series(
            _uint8_round_trip(image, 0.0, 1.0),
            len(clean),
            clean,
            metadata=metadata,
        )
    elif encoding == "rp":
        image = series_to_rp(clean, size=image_size)
        decoded = rp_to_series(
            _uint8_round_trip(image, 0.0, 1.0),
            len(clean),
            clean,
        )
    elif encoding == "spec":
        image, metadata = series_to_spectrogram(clean, size=image_size, return_metadata=True)
        decoded = spectrogram_to_series(
            _uint8_round_trip(image, float(np.min(image)), float(np.max(image))),
            len(clean),
            clean,
            metadata=metadata,
        )
    else:
        raise ValueError(f"Unknown SD2 encoding: {encoding!r}; expected one of {ENCODINGS}")

    return pd.Series(decoded.to_numpy(dtype=np.float64), index=clean.index)


def oracle_reconstruction(
    clean: pd.Series,
    degraded: pd.Series,
    encoding: str,
    image_size: int,
) -> pd.Series:
    """Fill missing positions from the decoded ideal clean image.

    This deliberately uses clean-image encoder metadata. It is an oracle upper
    bound for the representation and decoder, not a deployable imputer.
    """
    if not clean.index.equals(degraded.index):
        raise ValueError("Clean and degraded series must have identical indexes")

    decoded_clean = round_trip_series(clean, encoding, image_size)
    result = degraded.copy()
    missing = degraded.isna()
    result.loc[missing] = decoded_clean.loc[missing]
    return result


def run_ablation_cases(
    clean: pd.Series,
    encodings: list[str],
    image_size: int,
    mechanisms: dict[str, Callable[..., pd.Series]],
    rates: list[float],
    seed: int,
) -> list[dict]:
    """Evaluate one clean window under round-trip and oracle controls."""
    from reconstruction_metrics import (
        compute_metrics_from_series,
        compute_reconstruction_metrics,
    )

    rows: list[dict] = []
    for encoding in encodings:
        decoded = round_trip_series(clean, encoding, image_size)
        round_trip_metrics = compute_reconstruction_metrics(
            clean.to_numpy(dtype=np.float64),
            decoded.to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "ablation": "round_trip",
                "encoding": encoding,
                "image_size": image_size,
                "mechanism": "none",
                "missing_rate": 0.0,
                "scored_positions": len(clean),
                **round_trip_metrics,
            }
        )

        for case_number, (mechanism_name, mechanism) in enumerate(mechanisms.items()):
            rate = rates[case_number % len(rates)]
            degraded = mechanism(clean, rate, seed=seed + case_number)
            oracle = degraded.copy()
            missing = degraded.isna()
            oracle.loc[missing] = decoded.loc[missing]
            metrics = compute_metrics_from_series(clean, degraded, oracle)
            if metrics is None:
                continue
            rows.append(
                {
                    "ablation": "oracle_clean_image",
                    "encoding": encoding,
                    "image_size": image_size,
                    "mechanism": mechanism_name,
                    "missing_rate": rate,
                    "scored_positions": int(degraded.isna().sum()),
                    **metrics,
                }
            )
    return rows

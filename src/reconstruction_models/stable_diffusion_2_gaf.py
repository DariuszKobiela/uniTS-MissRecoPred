"""
Stable Diffusion 2 inpainting with GAF (Gramian Angular Field).

Two registry entries share this encoding:

- ``stable_diffusion_2_gaf`` — base weights
  (``stabilityai/stable-diffusion-2-inpainting``), matching the experiment
  ``*gafunet`` files.
- ``stable_diffusion_2_gaf_finetuned`` — UNet fine-tuned on GAF/MTF/RP/SPEC
  (``Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec``), matching
  ``*gafsd2all4``.
"""

from itertools import count

import numpy as np
import pandas as pd
from PIL import Image

from .sd2_pipeline import MODEL_ID_BASE, MODEL_ID_FINETUNED, get_model, seeded_generator
from .sd2_windowing import reconstruct_in_windows

DEFAULT_PROMPT = "high quality gramian angular field mathematical visualization"


def series_to_gaf(series: pd.Series, size: int = 512) -> np.ndarray:
    """Convert time series to Gramian Angular Field (GAF) image.

    Important: computing GAF is O(n^2). To avoid RAM OOM on long series, we first
    resample the series to `size` (default 512) and compute a fixed 512×512 GAF.
    """
    values = series.to_numpy(dtype=np.float32, copy=False)

    # Resample to fixed length to avoid O(n^2) blow-up
    if len(values) != size:
        x_old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=size, dtype=np.float32)
        values = np.interp(x_new, x_old, values).astype(np.float32, copy=False)

    min_val = float(np.min(values))
    max_val = float(np.max(values))

    if max_val - min_val > 0:
        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(values, dtype=np.float32)

    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
    phi = np.arccos(normalized)
    gaf = np.cos(phi[:, None] + phi[None, :]).astype(np.float32, copy=False)

    return gaf


def gaf_to_series(gaf: np.ndarray, original_length: int, original_series: pd.Series = None) -> pd.Series:
    """Invert a Gramian Angular Summation Field through its diagonal.

    For values scaled to ``[0, 1]``, ``G[i, i] = 2*x[i]**2 - 1``;
    therefore ``x[i] = sqrt((G[i, i] + 1) / 2)``. The previous
    ``cos(diagonal)`` operation was not an inverse of GASF.
    """
    diagonal = np.clip(np.diag(gaf).astype(np.float64), -1.0, 1.0)
    normalized = np.sqrt(np.clip((diagonal + 1.0) / 2.0, 0.0, 1.0))

    if len(normalized) != original_length:
        x_old = np.linspace(0.0, 1.0, len(normalized))
        x_new = np.linspace(0.0, 1.0, original_length)
        normalized = np.interp(x_new, x_old, normalized)

    if original_series is not None:
        reference = pd.to_numeric(original_series, errors="coerce").dropna()
        min_val = float(reference.min()) if len(reference) else 0.0
        max_val = float(reference.max()) if len(reference) else 1.0
        index = original_series.index
    else:
        min_val, max_val = 0.0, 1.0
        index = None

    if max_val > min_val:
        values = normalized * (max_val - min_val) + min_val
    else:
        values = np.full_like(normalized, min_val)

    return pd.Series(values, index=index)


def _inpaint_gaf_window(
    data: pd.Series,
    num_inference_steps: int,
    guidance_scale: float,
    model_id: str,
    variant: str,
    image_size: int,
    prompt: str,
    seed: int,
) -> pd.Series:
    """Run one local GAF inpainting task."""
    mask = data.isna()

    if not mask.any():
        return data.copy()

    # Fill NaN temporarily for encoding
    series_filled = data.interpolate(method="linear", limit_direction="both")
    if series_filled.isna().any():
        series_filled = series_filled.fillna(0)

    # Convert to GAF image
    print("  Converting time series to GAF image...")
    gaf_image = series_to_gaf(series_filled, size=image_size)

    # Create PIL images
    gaf_normalized = ((np.clip(gaf_image, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    image_pil = Image.fromarray(gaf_normalized).convert("RGB")

    # Create mask image (white = inpaint, black = keep)
    mask_2d = np.zeros_like(gaf_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = 0 if len(mask) == 1 else int(round(i * (gaf_image.shape[0] - 1) / (len(mask) - 1)))
            mask_2d[idx, :] = 1
            mask_2d[:, idx] = 1

    mask_normalized = (mask_2d * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized).convert("L")

    if image_pil.size != (image_size, image_size):
        image_pil = image_pil.resize((image_size, image_size))
        mask_pil = mask_pil.resize((image_size, image_size))

    pipeline = get_model(model_id)

    print(f"  Running Stable Diffusion 2 inpainting (GAF, {variant})...")
    print(f"    Steps: {num_inference_steps}, Guidance: {guidance_scale}")

    # Run inpainting
    result = pipeline(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=seeded_generator(pipeline, seed),
    ).images[0]

    # Convert result back
    result_array = np.array(result.convert("L"))
    gaf_reconstructed = result_array.astype(np.float64) / 127.5 - 1.0

    # Convert GAF back to time series
    print("  Converting GAF image back to time series...")
    reconstructed_series = gaf_to_series(gaf_reconstructed, len(data), series_filled)

    # Merge: keep original values, use reconstructed for missing
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]

    print(f"  Stable Diffusion 2 GAF ({variant}) inpainting completed")
    return result_series


def stable_diffusion_2_gaf(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
    seed: int = 42,
) -> pd.Series:
    """Impute missing values with base SD2 inpainting on local GAF images."""
    window_seeds = count(seed)
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_gaf_window(
            window,
            num_inference_steps,
            guidance_scale,
            MODEL_ID_BASE,
            "base",
            image_size,
            prompt or DEFAULT_PROMPT,
            next(window_seeds),
        ),
        window_samples,
        context_samples,
        progress_label="GAF",
    )


def stable_diffusion_2_gaf_finetuned(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
    seed: int = 42,
) -> pd.Series:
    """Impute missing values with fine-tuned SD2 on local GAF images."""
    window_seeds = count(seed)
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_gaf_window(
            window,
            num_inference_steps,
            guidance_scale,
            MODEL_ID_FINETUNED,
            "finetuned",
            image_size,
            prompt or DEFAULT_PROMPT,
            next(window_seeds),
        ),
        window_samples,
        context_samples,
        progress_label="GAF",
    )

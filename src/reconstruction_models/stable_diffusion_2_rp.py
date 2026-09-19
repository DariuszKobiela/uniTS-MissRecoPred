"""
Stable Diffusion 2 inpainting with RP (Recurrence Plot).

- ``stable_diffusion_2_rp`` — base SD2 (experiment ``*rpunet``).
- ``stable_diffusion_2_rp_finetuned`` — fine-tuned SD2 (experiment ``*rpsd2all4``).
"""

import numpy as np
import pandas as pd
from PIL import Image

from .sd2_pipeline import MODEL_ID_BASE, MODEL_ID_FINETUNED, get_model
from .sd2_windowing import reconstruct_in_windows

DEFAULT_PROMPT = "high quality continuous recurrence distance plot mathematical visualization"


def series_to_rp(series: pd.Series, epsilon: float = None, size: int = 512) -> np.ndarray:
    """Convert a series to a continuous recurrence-distance plot.

    A thresholded binary RP is not invertible. This variant stores normalized
    pairwise distances, which retain the one-dimensional geometry and can be
    inverted up to translation and reflection. ``epsilon`` is retained only for
    backwards API compatibility and is intentionally unused.
    """
    del epsilon
    values = series.to_numpy(dtype=np.float32, copy=False)

    if len(values) != size:
        x_old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=size, dtype=np.float32)
        values = np.interp(x_new, x_old, values).astype(np.float32, copy=False)

    min_val, max_val = values.min(), values.max()
    if max_val > min_val:
        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(values, dtype=np.float32)

    return np.abs(normalized[:, None] - normalized[None, :]).astype(np.float32, copy=False)


def rp_to_series(rp: np.ndarray, original_length: int, original_series: pd.Series = None) -> pd.Series:
    """Recover 1-D coordinates from a continuous pairwise-distance matrix."""
    distance = np.asarray(rp, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("RP must be a square distance matrix")

    # SD output need not be perfectly symmetric. Project it back onto the space
    # of symmetric non-negative dissimilarity matrices before reconstruction.
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    anchor_a, anchor_b = np.unravel_index(np.argmax(distance), distance.shape)
    span = float(distance[anchor_a, anchor_b])
    if span <= np.finfo(np.float64).eps:
        normalized = np.zeros(distance.shape[0], dtype=np.float64)
    else:
        # Trilateration on a line using the farthest pair as endpoints.
        normalized = (distance[anchor_a] ** 2 + span**2 - distance[anchor_b] ** 2) / (2.0 * span)
        normalized = np.clip(normalized / span, 0.0, 1.0)

    if original_series is not None:
        reference = pd.to_numeric(original_series, errors="coerce")
        reference = reference.interpolate(method="linear", limit_direction="both")
        reference_values = reference.to_numpy(dtype=np.float64)
        x_ref = np.linspace(0.0, 1.0, len(reference_values))
        x_rp = np.linspace(0.0, 1.0, len(normalized))
        reference_at_rp = np.interp(x_rp, x_ref, reference_values)

        corr = np.corrcoef(normalized, reference_at_rp)[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            normalized = 1.0 - normalized

        min_val = float(np.nanmin(reference_values))
        max_val = float(np.nanmax(reference_values))
        index = original_series.index
    else:
        min_val, max_val = 0.0, 1.0
        index = None

    if len(normalized) != original_length:
        x_old = np.linspace(0.0, 1.0, len(normalized))
        x_new = np.linspace(0.0, 1.0, original_length)
        normalized = np.interp(x_new, x_old, normalized)

    values = normalized * (max_val - min_val) + min_val
    return pd.Series(values, index=index)


def _inpaint_rp_window(
    data: pd.Series,
    num_inference_steps: int,
    guidance_scale: float,
    model_id: str,
    variant: str,
    image_size: int,
    prompt: str,
) -> pd.Series:
    mask = data.isna()

    if not mask.any():
        return data.copy()

    # Fill NaN temporarily
    series_filled = data.interpolate(method="linear", limit_direction="both")
    if series_filled.isna().any():
        series_filled = series_filled.fillna(0)

    # Convert to RP image
    print("  Converting time series to RP image...")
    rp_image = series_to_rp(series_filled, size=image_size)

    # Create PIL images
    rp_normalized = (np.clip(rp_image, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_pil = Image.fromarray(rp_normalized).convert("RGB")

    # Create mask image
    mask_2d = np.zeros_like(rp_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = 0 if len(mask) == 1 else int(round(i * (rp_image.shape[0] - 1) / (len(mask) - 1)))
            mask_2d[idx, :] = 1
            mask_2d[:, idx] = 1

    mask_normalized = (mask_2d * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized).convert("L")

    if image_pil.size != (image_size, image_size):
        image_pil = image_pil.resize((image_size, image_size))
        mask_pil = mask_pil.resize((image_size, image_size))

    pipeline = get_model(model_id)

    print(f"  Running Stable Diffusion 2 inpainting (RP, {variant})...")
    print(f"    Steps: {num_inference_steps}, Guidance: {guidance_scale}")

    # Run inpainting
    result = pipeline(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Convert result back
    result_array = np.array(result.convert("L"))
    rp_reconstructed = result_array.astype(np.float64) / 255.0

    # Convert RP back to time series
    print("  Converting RP image back to time series...")
    reconstructed_series = rp_to_series(rp_reconstructed, len(data), series_filled)

    # Merge
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]

    print(f"  Stable Diffusion 2 RP ({variant}) inpainting completed")
    return result_series


def stable_diffusion_2_rp(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
) -> pd.Series:
    """Impute missing values with base SD2 on local RP images."""
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_rp_window(
            window,
            num_inference_steps,
            guidance_scale,
            MODEL_ID_BASE,
            "base",
            image_size,
            prompt or DEFAULT_PROMPT,
        ),
        window_samples,
        context_samples,
        progress_label="RP",
    )


def stable_diffusion_2_rp_finetuned(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
) -> pd.Series:
    """Impute missing values with fine-tuned SD2 on local RP images."""
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_rp_window(
            window,
            num_inference_steps,
            guidance_scale,
            MODEL_ID_FINETUNED,
            "finetuned",
            image_size,
            prompt or DEFAULT_PROMPT,
        ),
        window_samples,
        context_samples,
        progress_label="RP",
    )

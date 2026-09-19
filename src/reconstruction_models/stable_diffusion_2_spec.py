"""
Stable Diffusion 2 inpainting with Spectrogram.

- ``stable_diffusion_2_spec`` — base SD2 (experiment ``*specunet``).
- ``stable_diffusion_2_spec_finetuned`` — fine-tuned SD2 (experiment ``*specsd2all4``).
"""

import numpy as np
import pandas as pd
from PIL import Image

from .sd2_pipeline import MODEL_ID_BASE, MODEL_ID_FINETUNED, get_model
from .sd2_windowing import reconstruct_in_windows

DEFAULT_PROMPT = "high quality spectrogram mathematical visualization"


def series_to_spectrogram(
    series: pd.Series,
    size: int = 512,
    return_metadata: bool = False,
):
    """Convert a series to a log-magnitude STFT image.

    When requested, metadata contains the phase and STFT parameters required for
    a mathematically valid inverse transform. The phase comes only from the
    linearly filled input supplied to the encoder, never from hidden truth.
    """
    from scipy import signal

    values = series.to_numpy(dtype=np.float64, copy=False)
    if len(values) < 2:
        raise ValueError("Spectrogram requires at least two samples")

    nperseg = min(256, max(2, len(values) // 4))
    noverlap = nperseg // 2
    _, _, stft = signal.stft(
        values,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(stft), 1e-8))
    original_shape = magnitude_db.shape

    if magnitude_db.shape != (size, size):
        from scipy.ndimage import zoom

        magnitude_db = zoom(
            magnitude_db,
            (size / original_shape[0], size / original_shape[1]),
            order=1,
        )

    if return_metadata:
        return magnitude_db, {
            "phase": np.angle(stft),
            "stft_shape": original_shape,
            "nperseg": nperseg,
            "noverlap": noverlap,
        }
    return magnitude_db


def spectrogram_to_series(
    spec: np.ndarray,
    original_length: int,
    original_series: pd.Series = None,
    metadata: dict | None = None,
) -> pd.Series:
    """Invert a log-magnitude STFT using its retained phase."""
    from scipy import signal

    if metadata is None:
        raise ValueError("Spectrogram decoding requires phase metadata returned by series_to_spectrogram")

    magnitude_db = np.asarray(spec, dtype=np.float64)
    target_shape = tuple(metadata["stft_shape"])
    if magnitude_db.shape != target_shape:
        from scipy.ndimage import zoom

        magnitude_db = zoom(
            magnitude_db,
            (target_shape[0] / magnitude_db.shape[0], target_shape[1] / magnitude_db.shape[1]),
            order=1,
        )

    magnitude = 10.0 ** (magnitude_db / 20.0)
    phase = np.asarray(metadata["phase"], dtype=np.float64)
    if phase.shape != magnitude.shape:
        raise ValueError("Stored STFT phase shape does not match decoded magnitude")

    complex_stft = magnitude * np.exp(1j * phase)
    _, values = signal.istft(
        complex_stft,
        nperseg=int(metadata["nperseg"]),
        noverlap=int(metadata["noverlap"]),
        input_onesided=True,
        boundary=True,
    )
    values = np.asarray(values, dtype=np.float64)
    if len(values) < original_length:
        values = np.pad(values, (0, original_length - len(values)), mode="edge")
    values = values[:original_length]

    index = original_series.index if original_series is not None else None
    return pd.Series(values, index=index)


def _inpaint_spec_window(
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

    # Convert to Spectrogram
    print("  Converting time series to Spectrogram...")
    spec_image, spec_metadata = series_to_spectrogram(series_filled, size=image_size, return_metadata=True)

    # Create PIL images
    spec_min = float(np.min(spec_image))
    spec_max = float(np.max(spec_image))
    if spec_max > spec_min:
        spec_normalized = ((spec_image - spec_min) / (spec_max - spec_min) * 255).astype(np.uint8)
    else:
        spec_normalized = np.zeros_like(spec_image, dtype=np.uint8)
    image_pil = Image.fromarray(spec_normalized).convert("RGB")

    # Create mask image
    mask_2d = np.zeros_like(spec_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = 0 if len(mask) == 1 else int(round(i * (spec_image.shape[1] - 1) / (len(mask) - 1)))
            mask_2d[:, idx] = 1  # Vertical stripe for time dimension

    mask_normalized = (mask_2d * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized).convert("L")

    if image_pil.size != (image_size, image_size):
        image_pil = image_pil.resize((image_size, image_size))
        mask_pil = mask_pil.resize((image_size, image_size))

    pipeline = get_model(model_id)

    print(f"  Running Stable Diffusion 2 inpainting (Spectrogram, {variant})...")
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
    spec_reconstructed = result_array.astype(float) / 255.0 * (spec_max - spec_min) + spec_min

    # Convert Spectrogram back to time series
    print("  Converting Spectrogram back to time series...")
    reconstructed_series = spectrogram_to_series(
        spec_reconstructed,
        len(data),
        data,
        metadata=spec_metadata,
    )

    # Merge
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]

    print(f"  Stable Diffusion 2 Spectrogram ({variant}) inpainting completed")
    return result_series


def stable_diffusion_2_spec(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
) -> pd.Series:
    """Impute missing values with base SD2 on local SPEC images."""
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_spec_window(
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
        progress_label="SPEC",
    )


def stable_diffusion_2_spec_finetuned(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
) -> pd.Series:
    """Impute missing values with fine-tuned SD2 on local SPEC images."""
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_spec_window(
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
        progress_label="SPEC",
    )

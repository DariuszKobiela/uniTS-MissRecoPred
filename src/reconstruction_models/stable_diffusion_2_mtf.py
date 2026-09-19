"""
Stable Diffusion 2 inpainting with MTF (Markov Transition Field).

- ``stable_diffusion_2_mtf`` — base SD2 (experiment ``*mtfunet``).
- ``stable_diffusion_2_mtf_finetuned`` — fine-tuned SD2 (experiment ``*mtfsd2all4``).
"""

from itertools import count

import numpy as np
import pandas as pd
from PIL import Image

from .sd2_pipeline import MODEL_ID_BASE, MODEL_ID_FINETUNED, get_model, seeded_generator
from .sd2_windowing import reconstruct_in_windows

DEFAULT_PROMPT = "high quality markov transition field mathematical visualization"


def series_to_mtf(
    series: pd.Series,
    n_bins: int = 8,
    size: int = 512,
    return_metadata: bool = False,
):
    """Convert a series to a genuine quantile-binned Markov Transition Field.

    ``MTF[i, j]`` contains the learned transition probability from the quantile
    state at time ``i`` to the state at time ``j``. The previous implementation
    only tested whether two samples occupied the same bin and was not an MTF.

    MTF quantization is inherently many-to-one. Optional metadata stores bin
    centres and the transition matrix for a conditional approximate decoder; it
    never contains unavailable ground-truth values.
    """
    values = series.to_numpy(dtype=np.float32, copy=False)

    if len(values) != size:
        x_old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=size, dtype=np.float32)
        values = np.interp(x_new, x_old, values).astype(np.float32, copy=False)

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        states = np.zeros(len(values), dtype=np.int64)
        centers = np.array([float(values[0])], dtype=np.float64)
    else:
        states = np.digitize(values, edges[1:-1], right=True).astype(np.int64)
        n_states = len(edges) - 1
        centers = np.empty(n_states, dtype=np.float64)
        for state in range(n_states):
            members = values[states == state]
            centers[state] = float(np.mean(members)) if len(members) else float((edges[state] + edges[state + 1]) / 2.0)

    n_states = len(centers)
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    if len(states) > 1:
        np.add.at(transition_counts, (states[:-1], states[1:]), 1.0)

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts),
        where=row_sums > 0.0,
    )
    for state in np.flatnonzero(row_sums.ravel() == 0.0):
        transition[state, state] = 1.0

    mtf = transition[states[:, None], states[None, :]].astype(np.float32)

    if return_metadata:
        return mtf, {
            "centers": centers,
            "transition": transition,
            "reference_states": states,
            "missing_encoded": np.zeros(len(states), dtype=bool),
        }

    return mtf


def mtf_to_series(
    mtf: np.ndarray,
    original_length: int,
    original_series: pd.Series = None,
    metadata: dict | None = None,
) -> pd.Series:
    """Approximately decode MTF states, conditional on observed time points.

    A pure MTF is not bijective. For every encoded missing position, this
    decoder selects the quantile state whose expected transition row and column
    best match the inpainted field at positions that remained observed.
    """
    if metadata is None:
        raise ValueError("MTF decoding requires metadata returned by series_to_mtf")

    field = np.clip(np.asarray(mtf, dtype=np.float64), 0.0, 1.0)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("MTF must be a square matrix")

    centers = np.asarray(metadata["centers"], dtype=np.float64)
    transition = np.asarray(metadata["transition"], dtype=np.float64)
    states = np.asarray(metadata["reference_states"], dtype=np.int64).copy()
    missing_encoded = np.asarray(metadata["missing_encoded"], dtype=bool)
    known = np.flatnonzero(~missing_encoded)

    if len(known):
        known_states = states[known]
        for idx in np.flatnonzero(missing_encoded):
            losses = np.empty(len(centers), dtype=np.float64)
            for candidate in range(len(centers)):
                row_error = field[idx, known] - transition[candidate, known_states]
                col_error = field[known, idx] - transition[known_states, candidate]
                losses[candidate] = np.mean(row_error**2) + np.mean(col_error**2)
            states[idx] = int(np.argmin(losses))

    decoded = centers[states]
    if len(decoded) != original_length:
        x_old = np.linspace(0.0, 1.0, len(decoded))
        x_new = np.linspace(0.0, 1.0, original_length)
        decoded = np.interp(x_new, x_old, decoded)

    index = original_series.index if original_series is not None else None
    return pd.Series(decoded, index=index)


def _inpaint_mtf_window(
    data: pd.Series,
    num_inference_steps: int,
    guidance_scale: float,
    model_id: str,
    variant: str,
    image_size: int,
    prompt: str,
    seed: int,
) -> pd.Series:
    mask = data.isna()

    if not mask.any():
        return data.copy()

    # Fill NaN temporarily
    series_filled = data.interpolate(method="linear", limit_direction="both")
    if series_filled.isna().any():
        series_filled = series_filled.fillna(0)

    # Convert to MTF image
    print("  Converting time series to MTF image...")
    mtf_image, mtf_metadata = series_to_mtf(series_filled, size=image_size, return_metadata=True)

    # Create PIL images
    mtf_normalized = (np.clip(mtf_image, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_pil = Image.fromarray(mtf_normalized).convert("RGB")

    # Create mask image
    mask_2d = np.zeros_like(mtf_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = 0 if len(mask) == 1 else int(round(i * (mtf_image.shape[0] - 1) / (len(mask) - 1)))
            mask_2d[idx, :] = 1
            mask_2d[:, idx] = 1
            mtf_metadata["missing_encoded"][idx] = True

    mask_normalized = (mask_2d * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized).convert("L")

    if image_pil.size != (image_size, image_size):
        image_pil = image_pil.resize((image_size, image_size))
        mask_pil = mask_pil.resize((image_size, image_size))

    pipeline = get_model(model_id)

    print(f"  Running Stable Diffusion 2 inpainting (MTF, {variant})...")
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
    mtf_reconstructed = result_array.astype(np.float64) / 255.0

    # Convert MTF back to time series
    print("  Converting MTF image back to time series...")
    reconstructed_series = mtf_to_series(
        mtf_reconstructed,
        len(data),
        data,
        metadata=mtf_metadata,
    )
    # Merge
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]

    print(f"  Stable Diffusion 2 MTF ({variant}) inpainting completed")
    return result_series


def stable_diffusion_2_mtf(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
    seed: int = 42,
) -> pd.Series:
    """Impute missing values with base SD2 on local MTF images."""
    window_seeds = count(seed)
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_mtf_window(
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
        progress_label="MTF",
    )


def stable_diffusion_2_mtf_finetuned(
    data: pd.Series,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    window_samples: int = 512,
    context_samples: int = 64,
    image_size: int = 512,
    prompt: str | None = None,
    seed: int = 42,
) -> pd.Series:
    """Impute missing values with fine-tuned SD2 on local MTF images."""
    window_seeds = count(seed)
    return reconstruct_in_windows(
        data,
        lambda window: _inpaint_mtf_window(
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
        progress_label="MTF",
    )

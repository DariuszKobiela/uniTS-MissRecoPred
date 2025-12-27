"""
Stable Diffusion 2 Inpainting with MTF (Markov Transition Field)
Uses fine-tuned Stable Diffusion 2 model for time series image inpainting.
Model: https://huggingface.co/Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec
"""

import pandas as pd
import numpy as np
import torch
import warnings
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


# Global model cache (shared across all SD2 models)
_MODEL_CACHE = {}


def get_model(model_id: str = "Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec"):
    """Load the Stable Diffusion 2 inpainting model. Uses cache to avoid reloading."""
    if model_id not in _MODEL_CACHE:
        print(f"Loading Stable Diffusion 2 model from HuggingFace: {model_id}")
        print("This may take a while on first run (downloading ~20GB model)...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Using device: {device}")
        
        # Note: safety_checker=None is OK for scientific research on time series data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*safety_checker.*')
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None
            ).to(device)
        
        if device == "cuda":
            pipeline.enable_attention_slicing()
        
        _MODEL_CACHE[model_id] = pipeline
        print(f"✓ Model loaded successfully and cached")
    
    return _MODEL_CACHE[model_id]


def series_to_mtf(series: pd.Series, n_bins: int = 8, size: int = 512) -> np.ndarray:
    """Convert time series to Markov Transition Field (MTF) image."""
    values = series.values
    
    # Quantize into bins
    min_val, max_val = values.min(), values.max()
    if max_val - min_val > 0:
        bins = np.linspace(min_val, max_val, n_bins + 1)
        quantized = np.digitize(values, bins[:-1]) - 1
        quantized = np.clip(quantized, 0, n_bins - 1)
    else:
        quantized = np.zeros(len(values), dtype=int)
    
    # Create transition matrix
    mtf = np.zeros((len(values), len(values)))
    for i in range(len(values) - 1):
        for j in range(i + 1, len(values)):
            transition_prob = 1.0 if quantized[i] == quantized[j] else 0.0
            mtf[i, j] = transition_prob
            mtf[j, i] = transition_prob
    
    # Resize if needed
    if mtf.shape[0] != size:
        from scipy.ndimage import zoom
        zoom_factor = size / mtf.shape[0]
        mtf = zoom(mtf, zoom_factor, order=1)
    
    return mtf


def mtf_to_series(mtf: np.ndarray, original_length: int, original_series: pd.Series = None) -> pd.Series:
    """Convert MTF image back to time series (approximate reconstruction)."""
    # Extract diagonal-like information
    diagonal = np.diag(mtf)
    
    # Resize to original length
    if len(diagonal) != original_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(diagonal))
        x_new = np.linspace(0, 1, original_length)
        f = interp1d(x_old, diagonal, kind='linear', bounds_error=False, fill_value='extrapolate')
        diagonal = f(x_new)
    
    # Scale back to original range
    if original_series is not None:
        min_val = original_series.min()
        max_val = original_series.max()
        if max_val - min_val > 0:
            values = diagonal * (max_val - min_val) + min_val
        else:
            values = np.full_like(diagonal, min_val)
    else:
        values = diagonal
    
    return pd.Series(values, index=original_series.index if original_series is not None else None)


def stable_diffusion_2_mtf(data: pd.Series,
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5) -> pd.Series:
    """
    Impute missing values using Stable Diffusion 2 inpainting with MTF encoding.
    
    Args:
        data: Pandas Series with missing values (NaN)
        num_inference_steps: Number of diffusion steps
        guidance_scale: Guidance scale for diffusion
        
    Returns:
        Pandas Series with missing values imputed
    """
    mask = data.isna()
    
    if not mask.any():
        return data.copy()
    
    # Fill NaN temporarily
    series_filled = data.interpolate(method='linear', limit_direction='both')
    if series_filled.isna().any():
        series_filled = series_filled.fillna(0)
    
    # Convert to MTF image
    print(f"  Converting time series to MTF image...")
    mtf_image = series_to_mtf(series_filled)
    
    # Create PIL images
    mtf_normalized = ((mtf_image - mtf_image.min()) / (mtf_image.max() - mtf_image.min()) * 255).astype(np.uint8)
    image_pil = Image.fromarray(mtf_normalized).convert("RGB")
    
    # Create mask image
    mask_2d = np.zeros_like(mtf_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = int(i * mtf_image.shape[0] / len(mask))
            mask_2d[idx, :] = 1
            mask_2d[:, idx] = 1
    
    mask_normalized = (mask_2d * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized).convert("L")
    
    # Resize to 512x512
    if image_pil.size != (512, 512):
        image_pil = image_pil.resize((512, 512))
        mask_pil = mask_pil.resize((512, 512))
    
    # Load model
    pipeline = get_model()
    
    # MTF-specific prompt
    prompt = "high quality markov transition field mathematical visualization"
    
    print(f"  Running Stable Diffusion 2 inpainting (MTF)...")
    print(f"    Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    # Run inpainting
    result = pipeline(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    # Convert result back
    result_array = np.array(result.convert("L"))
    mtf_reconstructed = (result_array.astype(float) / 255.0) * (mtf_image.max() - mtf_image.min()) + mtf_image.min()
    
    # Convert MTF back to time series
    print(f"  Converting MTF image back to time series...")
    reconstructed_series = mtf_to_series(mtf_reconstructed, len(data), data)
    
    # Merge
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]
    
    print(f"  ✓ Stable Diffusion 2 MTF inpainting completed")
    return result_series


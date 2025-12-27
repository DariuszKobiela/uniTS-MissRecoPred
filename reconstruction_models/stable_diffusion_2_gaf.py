"""
Stable Diffusion 2 Inpainting with GAF (Gramian Angular Field)
Uses fine-tuned Stable Diffusion 2 model for time series image inpainting.
Model: https://huggingface.co/Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec
"""

import pandas as pd
import numpy as np
import torch
import warnings
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from pathlib import Path


# Global model cache
_MODEL_CACHE = {}


def get_model(model_id: str = "Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec"):
    """
    Load the Stable Diffusion 2 inpainting model.
    Uses cache to avoid reloading.
    """
    if model_id not in _MODEL_CACHE:
        print(f"Loading Stable Diffusion 2 model from HuggingFace: {model_id}")
        print("This may take a while on first run (downloading ~20GB model)...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Using device: {device}")
        
        # Load model from HuggingFace (will use cache if already downloaded)
        # Note: safety_checker=None is OK for scientific research on time series data
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*safety_checker.*')
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None
            ).to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipeline.enable_attention_slicing()
        
        _MODEL_CACHE[model_id] = pipeline
        print(f"✓ Model loaded successfully and cached")
    
    return _MODEL_CACHE[model_id]


def series_to_gaf(series: pd.Series, size: int = 512) -> np.ndarray:
    """Convert time series to Gramian Angular Field (GAF) image."""
    values = series.values
    min_val, max_val = values.min(), values.max()
    
    if max_val - min_val > 0:
        normalized = 2 * (values - min_val) / (max_val - min_val) - 1
    else:
        normalized = np.zeros_like(values)
    
    normalized = np.clip(normalized, -1, 1)
    phi = np.arccos(normalized)
    gaf = np.cos(phi[:, None] + phi[None, :])
    
    if gaf.shape[0] != size:
        from scipy.ndimage import zoom
        zoom_factor = size / gaf.shape[0]
        gaf = zoom(gaf, zoom_factor, order=1)
    
    return gaf


def gaf_to_series(gaf: np.ndarray, original_length: int, original_series: pd.Series = None) -> pd.Series:
    """Convert GAF image back to time series."""
    diagonal = np.diag(gaf)
    normalized = np.cos(diagonal)
    
    if len(normalized) != original_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(normalized))
        x_new = np.linspace(0, 1, original_length)
        f = interp1d(x_old, normalized, kind='linear', bounds_error=False, fill_value='extrapolate')
        normalized = f(x_new)
    
    if original_series is not None:
        min_val = original_series.min()
        max_val = original_series.max()
    else:
        min_val, max_val = -1, 1
    
    if max_val - min_val > 0:
        values = (normalized + 1) / 2 * (max_val - min_val) + min_val
    else:
        values = np.full_like(normalized, min_val)
    
    return pd.Series(values, index=original_series.index if original_series is not None else None)


def stable_diffusion_2_gaf(data: pd.Series, 
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5) -> pd.Series:
    """
    Impute missing values using Stable Diffusion 2 inpainting with GAF encoding.
    
    Args:
        data: Pandas Series with missing values (NaN)
        num_inference_steps: Number of diffusion steps (higher = better quality, slower)
        guidance_scale: Guidance scale for diffusion (higher = closer to prompt)
        
    Returns:
        Pandas Series with missing values imputed
    """
    mask = data.isna()
    
    if not mask.any():
        return data.copy()
    
    # Fill NaN temporarily for encoding
    series_filled = data.interpolate(method='linear', limit_direction='both')
    if series_filled.isna().any():
        series_filled = series_filled.fillna(0)
    
    # Convert to GAF image
    print(f"  Converting time series to GAF image...")
    gaf_image = series_to_gaf(series_filled)
    
    # Create PIL images
    gaf_normalized = ((gaf_image - gaf_image.min()) / (gaf_image.max() - gaf_image.min()) * 255).astype(np.uint8)
    image_pil = Image.fromarray(gaf_normalized).convert("RGB")
    
    # Create mask image (white = inpaint, black = keep)
    mask_2d = np.zeros_like(gaf_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = int(i * gaf_image.shape[0] / len(mask))
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
    
    # GAF-specific prompt
    prompt = "high quality gramian angular field mathematical visualization"
    
    print(f"  Running Stable Diffusion 2 inpainting (GAF)...")
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
    gaf_reconstructed = (result_array.astype(float) / 255.0) * (gaf_image.max() - gaf_image.min()) + gaf_image.min()
    
    # Convert GAF back to time series
    print(f"  Converting GAF image back to time series...")
    reconstructed_series = gaf_to_series(gaf_reconstructed, len(data), data)
    
    # Merge: keep original values, use reconstructed for missing
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]
    
    print(f"  ✓ Stable Diffusion 2 GAF inpainting completed")
    return result_series


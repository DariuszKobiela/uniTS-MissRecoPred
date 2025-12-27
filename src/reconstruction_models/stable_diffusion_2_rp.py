"""
Stable Diffusion 2 Inpainting with RP (Recurrence Plot)
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


def series_to_rp(series: pd.Series, epsilon: float = None, size: int = 512) -> np.ndarray:
    """Convert time series to Recurrence Plot (RP) image."""
    values = series.values
    
    # Normalize values
    min_val, max_val = values.min(), values.max()
    if max_val - min_val > 0:
        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(values)
    
    # Calculate distance matrix
    diff = normalized[:, None] - normalized[None, :]
    distance = np.abs(diff)
    
    # Determine threshold if not provided
    if epsilon is None:
        epsilon = 0.1 * np.std(normalized)
    
    # Create recurrence plot (1 if distance < epsilon, 0 otherwise)
    rp = (distance < epsilon).astype(float)
    
    # Resize if needed
    if rp.shape[0] != size:
        from scipy.ndimage import zoom
        zoom_factor = size / rp.shape[0]
        rp = zoom(rp, zoom_factor, order=0)  # Nearest neighbor for binary
    
    return rp


def rp_to_series(rp: np.ndarray, original_length: int, original_series: pd.Series = None) -> pd.Series:
    """Convert RP image back to time series (approximate reconstruction)."""
    # Extract diagonal and nearby information
    diagonal = np.diag(rp)
    
    # Use row sums as proxy for value magnitude
    row_sums = np.sum(rp, axis=1) / rp.shape[1]
    
    # Resize to original length
    if len(row_sums) != original_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(row_sums))
        x_new = np.linspace(0, 1, original_length)
        f = interp1d(x_old, row_sums, kind='linear', bounds_error=False, fill_value='extrapolate')
        row_sums = f(x_new)
    
    # Scale back to original range
    if original_series is not None:
        min_val = original_series.min()
        max_val = original_series.max()
        if max_val - min_val > 0:
            values = row_sums * (max_val - min_val) + min_val
        else:
            values = np.full_like(row_sums, min_val)
    else:
        values = row_sums
    
    return pd.Series(values, index=original_series.index if original_series is not None else None)


def stable_diffusion_2_rp(data: pd.Series,
                          num_inference_steps: int = 50,
                          guidance_scale: float = 7.5) -> pd.Series:
    """
    Impute missing values using Stable Diffusion 2 inpainting with RP encoding.
    
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
    
    # Convert to RP image
    print(f"  Converting time series to RP image...")
    rp_image = series_to_rp(series_filled)
    
    # Create PIL images
    rp_normalized = ((rp_image - rp_image.min()) / (rp_image.max() - rp_image.min()) * 255).astype(np.uint8)
    image_pil = Image.fromarray(rp_normalized).convert("RGB")
    
    # Create mask image
    mask_2d = np.zeros_like(rp_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = int(i * rp_image.shape[0] / len(mask))
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
    
    # RP-specific prompt
    prompt = "high quality recurrence plot mathematical visualization"
    
    print(f"  Running Stable Diffusion 2 inpainting (RP)...")
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
    rp_reconstructed = (result_array.astype(float) / 255.0) * (rp_image.max() - rp_image.min()) + rp_image.min()
    
    # Convert RP back to time series
    print(f"  Converting RP image back to time series...")
    reconstructed_series = rp_to_series(rp_reconstructed, len(data), data)
    
    # Merge
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]
    
    print(f"  ✓ Stable Diffusion 2 RP inpainting completed")
    return result_series


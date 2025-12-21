"""
Stable Diffusion 2 Inpainting with Spectrogram
Uses fine-tuned Stable Diffusion 2 model for time series image inpainting.
Model: https://huggingface.co/Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec
"""

import pandas as pd
import numpy as np
import torch
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


def series_to_spectrogram(series: pd.Series, size: int = 512) -> np.ndarray:
    """Convert time series to Spectrogram image."""
    from scipy import signal
    
    values = series.values
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(values, nperseg=min(256, len(values)//4))
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Resize to square image
    if Sxx_db.shape[0] != size or Sxx_db.shape[1] != size:
        from scipy.ndimage import zoom
        zoom_factors = (size / Sxx_db.shape[0], size / Sxx_db.shape[1])
        Sxx_db = zoom(Sxx_db, zoom_factors, order=1)
    
    return Sxx_db


def spectrogram_to_series(spec: np.ndarray, original_length: int, original_series: pd.Series = None) -> pd.Series:
    """Convert Spectrogram back to time series (using inverse STFT approximation)."""
    from scipy import signal
    
    # Simple reconstruction: use column-wise average as amplitude proxy
    time_profile = np.mean(spec, axis=0)
    
    # Resize to original length
    if len(time_profile) != original_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(time_profile))
        x_new = np.linspace(0, 1, original_length)
        f = interp1d(x_old, time_profile, kind='linear', bounds_error=False, fill_value='extrapolate')
        time_profile = f(x_new)
    
    # Scale back to original range
    if original_series is not None:
        min_val = original_series.min()
        max_val = original_series.max()
        
        # Normalize time_profile
        tp_min, tp_max = time_profile.min(), time_profile.max()
        if tp_max - tp_min > 0:
            time_profile_norm = (time_profile - tp_min) / (tp_max - tp_min)
        else:
            time_profile_norm = np.zeros_like(time_profile)
        
        # Scale to original range
        if max_val - min_val > 0:
            values = time_profile_norm * (max_val - min_val) + min_val
        else:
            values = np.full_like(time_profile_norm, min_val)
    else:
        values = time_profile
    
    return pd.Series(values, index=original_series.index if original_series is not None else None)


def stable_diffusion_2_spec(data: pd.Series,
                            num_inference_steps: int = 50,
                            guidance_scale: float = 7.5) -> pd.Series:
    """
    Impute missing values using Stable Diffusion 2 inpainting with Spectrogram encoding.
    
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
    
    # Convert to Spectrogram
    print(f"  Converting time series to Spectrogram...")
    spec_image = series_to_spectrogram(series_filled)
    
    # Create PIL images
    spec_normalized = ((spec_image - spec_image.min()) / (spec_image.max() - spec_image.min()) * 255).astype(np.uint8)
    image_pil = Image.fromarray(spec_normalized).convert("RGB")
    
    # Create mask image
    mask_2d = np.zeros_like(spec_image)
    for i, is_missing in enumerate(mask):
        if is_missing:
            idx = int(i * spec_image.shape[1] / len(mask))
            mask_2d[:, idx] = 1  # Vertical stripe for time dimension
    
    mask_normalized = (mask_2d * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized).convert("L")
    
    # Resize to 512x512
    if image_pil.size != (512, 512):
        image_pil = image_pil.resize((512, 512))
        mask_pil = mask_pil.resize((512, 512))
    
    # Load model
    pipeline = get_model()
    
    # Spectrogram-specific prompt
    prompt = "high quality spectrogram mathematical visualization"
    
    print(f"  Running Stable Diffusion 2 inpainting (Spectrogram)...")
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
    spec_reconstructed = (result_array.astype(float) / 255.0) * (spec_image.max() - spec_image.min()) + spec_image.min()
    
    # Convert Spectrogram back to time series
    print(f"  Converting Spectrogram back to time series...")
    reconstructed_series = spectrogram_to_series(spec_reconstructed, len(data), data)
    
    # Merge
    result_series = data.copy()
    result_series[mask] = reconstructed_series[mask]
    
    print(f"  ✓ Stable Diffusion 2 Spectrogram inpainting completed")
    return result_series


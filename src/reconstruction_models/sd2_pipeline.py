"""
Shared Stable Diffusion 2 inpainting loader.

Two weights are used across GAF / MTF / RP / Spectrogram encodings:

- Base (off-the-shelf): local snapshot at ``models_cache/sd2_inpainting_base``
  (unzipped from ``sd2_inpainting_base.zip``). If missing, falls back to
  ``stabilityai/stable-diffusion-2-inpainting`` on the Hub (gated).
- Fine-tuned on all four encodings:
  ``Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec``
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_BASE_DIR = _REPO_ROOT / "models_cache" / "sd2_inpainting_base"
_DEFAULT_LOCAL_FINETUNED_DIR = _REPO_ROOT / "models" / "sd2_windowed" / "best_model"

MODEL_ID_HF_BASE = "stabilityai/stable-diffusion-2-inpainting"
MODEL_ID_HF_FINETUNED = "Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec"

# Prefer local full-pipeline snapshots and fall back to Hugging Face identifiers.
MODEL_ID_BASE = str(_LOCAL_BASE_DIR) if (_LOCAL_BASE_DIR / "model_index.json").is_file() else MODEL_ID_HF_BASE
_finetuned_override = Path(os.environ.get("SD2_FINETUNED_MODEL", str(_DEFAULT_LOCAL_FINETUNED_DIR)))
MODEL_ID_FINETUNED = (
    str(_finetuned_override) if (_finetuned_override / "model_index.json").is_file() else MODEL_ID_HF_FINETUNED
)

_MODEL_CACHE: dict[str, StableDiffusionInpaintPipeline] = {}


def get_model(model_id: str) -> StableDiffusionInpaintPipeline:
    """Load a Stable Diffusion 2 inpainting pipeline (cached per model_id)."""
    if model_id not in _MODEL_CACHE:
        local_dir = Path(model_id)
        is_local = local_dir.is_dir() and (local_dir / "model_index.json").is_file()

        print("Loading model... please wait")
        if is_local:
            print(f"Loading Stable Diffusion 2 inpainting weights from {local_dir}")
        else:
            print(f"Loading Stable Diffusion 2 model from HuggingFace: {model_id}")
            print("This may take a while on first run (downloading weights)...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Using device: {device}")
        print("  Notice: Safety checker is disabled for scientific time series reconstruction.")
        load_kwargs = dict(
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True,
        )
        if is_local:
            load_kwargs["local_files_only"] = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*safety_checker.*")
            warnings.filterwarnings("ignore", message=".*dtype.*")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                **load_kwargs,
            )

        if device == "cuda":
            pipeline = pipeline.to(device)
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()

        _MODEL_CACHE[model_id] = pipeline
        print("Model loaded successfully and cached")

    return _MODEL_CACHE[model_id]


def clear_model_cache() -> None:
    """Drop cached pipelines (used by hyperparameter search)."""
    _MODEL_CACHE.clear()

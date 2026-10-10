"""Resolve dataset- and encoding-specific SD2 runtime settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = {
    "gaf": "high quality gramian angular field mathematical visualization",
    "mtf": "high quality markov transition field mathematical visualization",
    "rp": "high quality continuous recurrence distance plot mathematical visualization",
    "spec": "high quality spectrogram mathematical visualization",
}


def encoding_from_model_name(model_name: str) -> str:
    for encoding in DEFAULT_PROMPTS:
        if f"_{encoding}" in model_name:
            return encoding
    raise ValueError(f"Cannot infer image encoding from model name: {model_name}")


def _dataset_override(mapping: dict[str, Any], dataset: str) -> Any:
    if dataset in mapping:
        return mapping[dataset]
    dataset_lower = dataset.lower()
    for key, value in mapping.items():
        if str(key).lower() in dataset_lower:
            return value
    return None

def _empirical_override(
    settings: dict[str, Any],
    dataset: str,
    model_name: str,
    encoding: str,
) -> dict[str, Any]:
    override_path = settings.get("runtime_overrides_path")
    if not override_path or not Path(override_path).is_file():
        return {}
    payload = json.loads(Path(override_path).read_text(encoding="utf-8"))
    dataset_overrides = _dataset_override(payload.get("overrides", {}), dataset)
    if not isinstance(dataset_overrides, dict):
        return {}
    override = dataset_overrides.get(model_name, dataset_overrides.get(encoding, {}))
    return override if isinstance(override, dict) else {}



def resolve_sd2_runtime_settings(
    settings: dict[str, Any],
    dataset: str,
    model_name: str,
    series_length: int,
) -> dict[str, Any]:
    """Return kwargs accepted by all built-in SD2 reconstruction functions."""
    encoding = encoding_from_model_name(model_name)
    windowing = settings.get("windowing", {}) or {}
    empirical = _empirical_override(settings, dataset, model_name, encoding)
    enabled = bool(windowing.get("enabled", True))

    default_window = int(windowing.get("default_window_samples", 512))
    override = _dataset_override(windowing.get("by_dataset", {}) or {}, dataset)
    window_samples = int(empirical.get("window_samples", override if override is not None else default_window))
    context_samples = int(empirical.get("context_samples", windowing.get("context_samples", 64)))

    if not enabled:
        window_samples = max(2, series_length)
        context_samples = 0
    elif series_length <= window_samples:
        context_samples = min(context_samples, max(0, (series_length - 1) // 2))

    if window_samples < 2:
        raise ValueError("Stable Diffusion window_samples must be >= 2")
    if context_samples < 0 or 2 * context_samples >= window_samples:
        raise ValueError("Stable Diffusion context_samples must be non-negative and smaller than half the window")

    image_sizes = settings.get("image_size_by_dataset", {}) or {}
    image_override = _dataset_override(image_sizes, dataset)
    image_size = int(empirical.get("image_size", image_override or settings.get("image_size", 512)))
    if image_size < 64 or image_size % 8:
        raise ValueError("Stable Diffusion image_size must be >= 64 and divisible by 8")

    prompts = settings.get("prompts", {}) or {}
    prompt = str(empirical.get("prompt", prompts.get(encoding, DEFAULT_PROMPTS[encoding])))

    return {
        "seed": int(empirical.get("seed", settings.get("seed", 42))),
        "num_inference_steps": int(empirical.get("num_inference_steps", settings.get("num_inference_steps", 42))),
        "guidance_scale": float(empirical.get("guidance_scale", settings.get("guidance_scale", 7.5))),
        "window_samples": window_samples,
        "context_samples": context_samples,
        "image_size": image_size,
        "prompt": prompt,
    }

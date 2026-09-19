import json

import numpy as np
import pandas as pd
import pytest

from reconstruction_models.sd2_settings import resolve_sd2_runtime_settings
from reconstruction_models.sd2_windowing import (
    plan_reconstruction_windows,
    reconstruct_in_windows,
)


def test_window_plan_has_disjoint_complete_cores_and_bounded_windows():
    plans = plan_reconstruction_windows(1000, window_samples=512, context_samples=64)

    assert [(p.core_start, p.core_stop) for p in plans] == [
        (0, 384),
        (384, 768),
        (768, 1000),
    ]
    assert all(p.window_length == 512 for p in plans)
    assert all(p.window_start <= p.core_start < p.core_stop <= p.window_stop for p in plans)


def test_windowed_reconstruction_writes_only_missing_core_values():
    source = pd.Series(np.arange(1000, dtype=float))
    degraded = source.copy()
    degraded.iloc[[10, 383, 384, 700, 999]] = np.nan
    calls = []

    def fake_reconstructor(window: pd.Series) -> pd.Series:
        calls.append((window.index[0], window.index[-1]))
        return window.fillna(pd.Series(source, index=source.index))

    reconstructed = reconstruct_in_windows(degraded, fake_reconstructor, window_samples=512, context_samples=64)

    pd.testing.assert_series_equal(reconstructed, source)
    assert len(calls) == 3


def test_windowed_reconstruction_validates_context():
    with pytest.raises(ValueError, match="smaller than half"):
        plan_reconstruction_windows(100, window_samples=64, context_samples=32)


def test_runtime_settings_resolve_dataset_and_prompt_overrides():
    settings = {
        "num_inference_steps": 30,
        "guidance_scale": 4.5,
        "image_size": 512,
        "windowing": {
            "enabled": True,
            "default_window_samples": 512,
            "context_samples": 64,
            "by_dataset": {"pump": 2048},
        },
        "prompts": {"gaf": "custom gaf prompt"},
    }

    runtime = resolve_sd2_runtime_settings(settings, "pump_sensor_28_univ", "stable_diffusion_2_gaf", 218880)

    assert runtime == {
        "seed": 42,
        "num_inference_steps": 30,
        "guidance_scale": 4.5,
        "window_samples": 2048,
        "context_samples": 64,
        "image_size": 512,
        "prompt": "custom gaf prompt",
    }


def test_runtime_settings_reject_invalid_image_size():
    settings = {"image_size": 510, "windowing": {"enabled": True}}
    with pytest.raises(ValueError, match="divisible by 8"):
        resolve_sd2_runtime_settings(settings, "boiler", "stable_diffusion_2_rp", 1000)


def test_runtime_settings_prefer_empirical_model_override(tmp_path):
    override_path = tmp_path / "runtime.json"
    override_path.write_text(
        json.dumps(
            {
                "overrides": {
                    "pump_sensor_28_univ": {
                        "stable_diffusion_2_gaf": {
                            "window_samples": 1024,
                            "image_size": 512,
                            "num_inference_steps": 30,
                            "guidance_scale": 3.0,
                            "prompt": "empirical prompt",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "runtime_overrides_path": str(override_path),
        "num_inference_steps": 42,
        "guidance_scale": 7.5,
        "image_size": 1024,
        "windowing": {"default_window_samples": 512, "context_samples": 64},
    }

    runtime = resolve_sd2_runtime_settings(
        settings,
        "pump_sensor_28_univ",
        "stable_diffusion_2_gaf",
        10_000,
    )

    assert runtime == {
        "seed": 42,
        "num_inference_steps": 30,
        "guidance_scale": 3.0,
        "window_samples": 1024,
        "context_samples": 64,
        "image_size": 512,
        "prompt": "empirical prompt",
    }

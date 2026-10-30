import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reconstruction_models.sd2_settings import resolve_sd2_runtime_settings
from reconstruction_models.sd2_windowing import (
    plan_reconstruction_windows,
    reconstruct_in_windows,
)


def test_hpo_missingness_guarantees_a_measurable_short_case():
    script = Path(__file__).parents[1] / "src" / "4_analyze_sd2_design.py"
    spec = importlib.util.spec_from_file_location("analyze_sd2_design", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clean = pd.Series(np.arange(24, dtype=float))

    degraded = module.apply_hpo_missingness(clean, "MCAR", 0.02, seed=42)

    assert degraded.isna().sum() == 1

    cases = module.build_hpo_cases([0.02, 0.05, 0.20, 0.50], 1)
    assert len(cases) == 12
    assert {(mechanism, rate) for mechanism, rate, _ in cases} == {
        (mechanism, rate) for mechanism in ("MCAR", "MAR", "MNAR") for rate in (0.02, 0.05, 0.20, 0.50)
    }

    assert module.effective_context_samples(64, len(clean)) == 11
    assert module.effective_context_samples(64, 500) == 64


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


def test_finetuned_model_inherits_zero_shot_empirical_override(tmp_path):
    override_path = tmp_path / "runtime.json"
    override_path.write_text(
        json.dumps(
            {
                "overrides": {
                    "vibration_sensor_S1": {
                        "stable_diffusion_2_gaf": {
                            "window_samples": 1024,
                            "image_size": 512,
                            "num_inference_steps": 30,
                            "guidance_scale": 3.0,
                            "prompt": "winning zero-shot prompt",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "runtime_overrides_path": str(override_path),
        "windowing": {"default_window_samples": 512, "context_samples": 64},
    }

    runtime = resolve_sd2_runtime_settings(
        settings,
        "vibration_sensor_S1",
        "stable_diffusion_2_gaf_finetuned",
        10_000,
    )

    assert runtime["window_samples"] == 1024
    assert runtime["num_inference_steps"] == 30
    assert runtime["guidance_scale"] == 3.0
    assert runtime["prompt"] == "winning zero-shot prompt"


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

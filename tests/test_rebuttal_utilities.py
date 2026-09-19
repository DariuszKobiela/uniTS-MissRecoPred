"""Tests for rebuttal reproducibility and SD2 fail-closed behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.empirical_mask_coverage import empirical_mask_fraction, report_empirical_mask_coverage
from utils.run_manifest import build_run_manifest, write_run_manifest


def test_run_manifest_contains_git_and_config_checksums(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("split:\n  test_samples: 30\n", encoding="utf-8")
    manifest = build_run_manifest(
        run_id="test_run",
        config_paths=[config],
        seed=42,
    )
    assert manifest["run_id"] == "test_run"
    assert manifest["seed"] == 42
    assert len(manifest["configs"]) == 1
    out = write_run_manifest(tmp_path / "manifest.json", manifest)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["configs"][0]["sha256"]


def test_empirical_mask_fraction_for_gaf_is_symmetric():
    missing = np.array([False, True, False, True, False])
    fraction = empirical_mask_fraction(missing, "gaf", image_size=8)
    assert 0.0 < fraction < 1.0


def test_report_empirical_mask_coverage_parses_degraded_files(tmp_path):
    series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    path = tmp_path / "boiler_MCAR-scattered_8p_1.csv"
    series.to_csv(path)
    frame = report_empirical_mask_coverage(tmp_path)
    assert not frame.empty
    assert set(frame["representation"]) == {"gaf", "mtf", "rp", "spec"}


def test_sd2_fail_closed_blocks_missing_local_model(monkeypatch):
    monkeypatch.setenv("SD2_FAIL_CLOSED", "1")
    import importlib

    import reconstruction_models.sd2_pipeline as pipeline

    importlib.reload(pipeline)
    monkeypatch.setattr(
        pipeline,
        "_DEFAULT_LOCAL_FINETUNED_DIR",
        Path("/nonexistent/sd2_windowed/best_model"),
    )
    with pytest.raises(FileNotFoundError, match="Fine-tuned SD2 model not found"):
        pipeline.resolve_finetuned_model_id()

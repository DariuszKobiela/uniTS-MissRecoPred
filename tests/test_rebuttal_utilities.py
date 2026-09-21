"""Tests for empirical masks and SD2 fail-closed behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.empirical_mask_coverage import empirical_mask_fraction, report_empirical_mask_coverage


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
    monkeypatch.setattr(pipeline, "_DEFAULT_LOCAL_FINETUNED_DIR", Path("/nonexistent/sd2_windowed/best_model"))
    with pytest.raises(FileNotFoundError, match="Fine-tuned SD2 model not found"):
        pipeline.resolve_finetuned_model_id()

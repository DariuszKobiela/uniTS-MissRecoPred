"""Tests for mechanism × temporal-structure missingness and gap reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missingness_techniques.structured import apply_structured_missingness
from utils.experiment_naming import decode_missingness_label, encode_missingness_label
from utils.missingness_analysis import summarize_missingness


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR"])
@pytest.mark.parametrize("structure", ["scattered", "contiguous", "mixed"])
def test_structured_realizations_are_exact_and_reproducible(mechanism, structure):
    source = pd.Series(np.sin(np.linspace(0, 20, 1000)))
    first = apply_structured_missingness(
        source, 0.2, mechanism, structure, seed=123
    )
    second = apply_structured_missingness(
        source, 0.2, mechanism, structure, seed=123
    )

    assert first.isna().sum() == 200
    assert first.isna().equals(second.isna())
    assert not source.isna().any()


def test_contiguous_has_no_singleton_gaps():
    result = apply_structured_missingness(
        pd.Series(np.arange(500, dtype=float)),
        0.2,
        "MCAR",
        "contiguous",
        seed=4,
        min_block_length=3,
        max_block_length=12,
    )
    summary, gaps = summarize_missingness(result)
    assert summary["singleton_gap_percent"] == 0.0
    assert min(gap["length_samples"] for gap in gaps) >= 3


def test_mixed_contains_points_and_blocks():
    result = apply_structured_missingness(
        pd.Series(np.arange(1000, dtype=float)),
        0.2,
        "MCAR",
        "mixed",
        seed=9,
        mixed_scattered_fraction=0.5,
    )
    _, gaps = summarize_missingness(result)
    lengths = [gap["length_samples"] for gap in gaps]
    assert 1 in lengths
    assert any(length > 1 for length in lengths)


def test_gap_report_contains_sample_and_time_statistics():
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    series = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0], index=index)
    summary, gaps = summarize_missingness(series)

    assert summary["n_gaps"] == 2
    assert summary["gap_length_samples_mean"] == pytest.approx(1.5)
    assert summary["gap_length_samples_p90"] == pytest.approx(1.9)
    assert summary["gap_coverage_seconds_max"] == pytest.approx(1800.0)
    assert summary["singleton_gap_percent"] == pytest.approx(50.0)
    assert gaps[1]["coverage_seconds"] == pytest.approx(900.0)


def test_missingness_filename_label_is_backward_compatible():
    assert encode_missingness_label("MNAR", "contiguous") == "MNAR-contiguous"
    assert decode_missingness_label("MNAR-contiguous") == ("MNAR", "contiguous")
    assert decode_missingness_label("MCAR") == ("MCAR", "scattered")


def test_step3_writes_all_structures_and_reports(tmp_path):
    script = Path(__file__).resolve().parents[1] / "src" / "8_degrade_datasets.py"
    spec = importlib.util.spec_from_file_location("degrade_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source = tmp_path / "sensor.csv"
    output = tmp_path / "missing"
    pd.DataFrame(
        {"value": np.sin(np.linspace(0, 10, 200))},
        index=pd.date_range("2025-01-01", periods=200, freq="5min"),
    ).to_csv(source)

    class Config:
        def get_datasets(self):
            return [str(source)]

        def get_missingness_techniques(self):
            return ["MCAR"]

        def get_missingness_rates(self):
            return [0.1]

        def get_missingness_structures(self):
            return ["scattered", "contiguous", "mixed"]

        def get_missingness_structure_settings(self):
            return {
                "min_block_length": 3,
                "max_block_length": 12,
                "mixed_scattered_fraction": 0.5,
            }

        def get_iterations(self):
            return 1

        def get_seed(self):
            return 42

        def get_missing_dir(self):
            return str(output)

        def get_n_jobs(self):
            return 1

        def get_csv_format(self, _filename):
            return {"index_col": 0}

    assert module.run_degrade_datasets(Config(), force=True)
    generated = sorted(output.glob("*.csv"))
    assert len(generated) == 3
    assert {path.stem.split("_")[-3] for path in generated} == {
        "MCAR-scattered",
        "MCAR-contiguous",
        "MCAR-mixed",
    }

    realizations = pd.read_csv(output / "reports" / "missingness_realizations.csv")
    gaps = pd.read_csv(output / "reports" / "missingness_gaps.csv")
    assert set(realizations["structure"]) == {"scattered", "contiguous", "mixed"}
    assert (realizations["n_missing"] == 20).all()
    assert {
        "n_gaps",
        "gap_length_samples_mean",
        "gap_length_samples_median",
        "gap_length_samples_p90",
        "gap_length_samples_max",
        "singleton_gap_percent",
        "gap_coverage_seconds_mean",
    }.issubset(realizations.columns)
    assert (gaps["coverage_seconds"] > 0).all()

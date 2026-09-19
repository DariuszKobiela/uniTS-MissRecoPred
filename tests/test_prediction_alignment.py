"""Strict prediction/test horizon alignment regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "9_calculate_prediction_error.py"
SPEC = importlib.util.spec_from_file_location("prediction_error_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
align_actual_predicted = MODULE.align_actual_predicted


def test_exact_datetime_index_aligns_after_csv_style_roundtrip():
    actual_index = pd.date_range("2025-01-01", periods=3, freq="h")
    predicted_index = pd.Index([str(value) for value in actual_index])
    actual = pd.Series([1.0, 2.0, 3.0], index=actual_index)
    predicted = pd.Series([1.1, 2.1, 3.1], index=predicted_index)

    y_true, y_pred = align_actual_predicted(actual, predicted)
    assert y_true.tolist() == [1.0, 2.0, 3.0]
    assert y_pred.tolist() == [1.1, 2.1, 3.1]


def test_length_mismatch_is_not_silently_truncated():
    with pytest.raises(ValueError, match="refusing to truncate"):
        align_actual_predicted(
            pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0])
        )


def test_shifted_or_reordered_index_is_rejected():
    actual = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    with pytest.raises(ValueError, match="index mismatch"):
        align_actual_predicted(
            actual, pd.Series([1.0, 2.0, 3.0], index=[11, 12, 13])
        )
    with pytest.raises(ValueError, match="index mismatch"):
        align_actual_predicted(
            actual, pd.Series([1.0, 2.0, 3.0], index=[10, 12, 11])
        )


def test_duplicate_index_and_nan_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        align_actual_predicted(
            pd.Series([1.0, 2.0], index=[0, 1]),
            pd.Series([1.0, 2.0], index=[0, 0]),
        )
    with pytest.raises(ValueError, match="finite"):
        align_actual_predicted(
            pd.Series([1.0, 2.0], index=[0, 1]),
            pd.Series([1.0, None], index=[0, 1]),
        )

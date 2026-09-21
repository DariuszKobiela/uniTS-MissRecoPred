"""Regression tests for explicit source-data forward filling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "1_clean_datasets.py"
SPEC = importlib.util.spec_from_file_location("clean_datasets_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_clean_dataset_forward_fills_and_reports(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "cleaned.csv"
    pd.DataFrame({"time": [0, 1, 2, 3], "value": [1.0, None, None, 4.0]}).to_csv(source, index=False)
    config = MagicMock()
    config.get_overwrite_existing.return_value = True

    report = MODULE.clean_dataset(str(source), str(output), config)
    cleaned = pd.read_csv(output, index_col=0)

    assert cleaned["value"].tolist() == [1.0, 1.0, 1.0, 4.0]
    assert report["missing_values_before"] == 2
    assert report["values_forward_filled"] == 2
    assert report["unresolved_rows_removed"] == 0


def test_clean_dataset_removes_only_unfillable_leading_gap(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "cleaned.csv"
    pd.DataFrame({"time": [0, 1, 2], "value": [None, 2.0, None]}).to_csv(source, index=False)
    config = MagicMock()
    config.get_overwrite_existing.return_value = True

    report = MODULE.clean_dataset(str(source), str(output), config)
    cleaned = pd.read_csv(output, index_col=0)

    assert cleaned["value"].tolist() == [2.0, 2.0]
    assert report["values_forward_filled"] == 1
    assert report["unresolved_rows_removed"] == 1

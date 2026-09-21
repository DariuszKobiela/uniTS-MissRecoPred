import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_path = Path(__file__).resolve().parents[1] / "src" / "6_analyze_synthetic_real_gap.py"
_spec = importlib.util.spec_from_file_location("analyze_synthetic_real_gap", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
FEATURE_COLUMNS = _module.FEATURE_COLUMNS
classifier_diagnostics = _module.classifier_diagnostics
extract_features = _module.extract_features


def test_feature_extraction_covers_distribution_acf_spectrum_and_changes():
    time = np.arange(256)
    values = 0.02 * time + np.sin(2 * np.pi * time / 16)
    features = extract_features(values)

    assert set(features) == set(FEATURE_COLUMNS)
    assert features["length"] == 256
    assert features["trend_slope"] > 0
    assert np.isclose(features["dominant_frequency"], 1 / 16)
    assert 0 <= features["spectral_entropy"] <= 1
    assert features["acf_1"] > 0
    assert features["diff_abs_mean"] > 0


def test_grouped_classifier_reports_separability():
    rng = np.random.default_rng(42)
    rows = []
    for source_kind, location in (("synthetic", -3.0), ("real", 3.0)):
        for group in range(4):
            for sample in range(5):
                row = {
                    "source_kind": source_kind,
                    "source_name": f"{source_kind}_{group}",
                    "series_id": f"{source_kind}_{group}_{sample}",
                }
                row.update(
                    {
                        feature: location + rng.normal(0, 0.1)
                        for feature in FEATURE_COLUMNS
                    }
                )
                rows.append(row)

    result = classifier_diagnostics(pd.DataFrame(rows), seed=42)
    assert result["status"] == "ok"
    assert result["roc_auc"] > 0.95
    assert result["balanced_accuracy"] > 0.95

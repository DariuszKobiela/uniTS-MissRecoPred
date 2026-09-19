"""Tests for batch statistical export pairing keys."""

from __future__ import annotations

import pandas as pd

from utils.statistical_tests import friedman_test, pairwise_comparisons


def test_rolling_origin_pairing_includes_origin_and_horizon():
    frame = pd.DataFrame(
        [
            {
                "dataset_name": "boiler",
                "technique": "MCAR",
                "structure": "scattered",
                "rate_percent": 8,
                "reconstruction_iteration": 1,
                "reconstruction_model": "interpolate_linear",
                "source_type": "reconstructed",
                "forecast_horizon": 12,
                "metric_scope": "cumulative",
                "origin": 1,
                "prediction_model": "xgboost",
                "model": "xgboost",
                "mae": 0.5,
            },
            {
                "dataset_name": "boiler",
                "technique": "MCAR",
                "structure": "scattered",
                "rate_percent": 8,
                "reconstruction_iteration": 1,
                "reconstruction_model": "interpolate_linear",
                "source_type": "reconstructed",
                "forecast_horizon": 12,
                "metric_scope": "cumulative",
                "origin": 1,
                "prediction_model": "sarimax",
                "model": "sarimax",
                "mae": 0.7,
            },
            {
                "dataset_name": "boiler",
                "technique": "MCAR",
                "structure": "scattered",
                "rate_percent": 8,
                "reconstruction_iteration": 1,
                "reconstruction_model": "interpolate_linear",
                "source_type": "reconstructed",
                "forecast_horizon": 12,
                "metric_scope": "cumulative",
                "origin": 2,
                "prediction_model": "xgboost",
                "model": "xgboost",
                "mae": 0.6,
            },
            {
                "dataset_name": "boiler",
                "technique": "MCAR",
                "structure": "scattered",
                "rate_percent": 8,
                "reconstruction_iteration": 1,
                "reconstruction_model": "interpolate_linear",
                "source_type": "reconstructed",
                "forecast_horizon": 12,
                "metric_scope": "cumulative",
                "origin": 2,
                "prediction_model": "sarimax",
                "model": "sarimax",
                "mae": 0.8,
            },
        ]
    )
    pair_columns = [
        "dataset_name",
        "technique",
        "structure",
        "rate_percent",
        "reconstruction_iteration",
        "reconstruction_model",
        "source_type",
        "forecast_horizon",
        "metric_scope",
        "origin",
    ]
    comparisons = pairwise_comparisons(frame, metric="mae", pair_columns=pair_columns)
    assert not comparisons.empty
    assert comparisons.iloc[0]["n_exact_pairs"] == 2

    omnibus = friedman_test(frame, metric="mae", pair_columns=pair_columns)
    assert omnibus["n_models"] == 2

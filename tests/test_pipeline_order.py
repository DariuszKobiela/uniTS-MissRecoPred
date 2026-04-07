"""run_pipeline_full step order (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import framework.runs as runs


@pytest.fixture
def mock_configs():
    return MagicMock(), MagicMock()


def test_run_pipeline_full_invokes_steps_in_order(mock_configs):
    config, pred = mock_configs
    order: list[str] = []

    def track(name):
        def _fn(*_a, **_k):
            order.append(name)
            return True

        return _fn

    with patch.multiple(
        runs,
        run_clean_datasets=track("clean"),
        run_create_split=track("split"),
        run_degrade_datasets=track("degrade"),
        run_reconstruct_datasets=track("reconstruct"),
        run_calculate_reconstruction_error=track("recon_err"),
        run_train_prediction_models=track("train"),
        run_predict_datasets=track("predict"),
        run_calculate_prediction_error=track("pred_err"),
        load_prediction_models_config=MagicMock(return_value=pred),
    ):
        result = runs.run_pipeline_full(config, pred_config=None)

    assert result.ok
    assert order == [
        "clean",
        "split",
        "degrade",
        "reconstruct",
        "recon_err",
        "train",
        "predict",
        "pred_err",
    ]


def test_run_pipeline_full_stops_on_first_failure(mock_configs):
    config, pred = mock_configs

    with patch.multiple(
        runs,
        run_clean_datasets=MagicMock(return_value=False),
        run_create_split=MagicMock(return_value=True),
        run_degrade_datasets=MagicMock(return_value=True),
        run_reconstruct_datasets=MagicMock(return_value=True),
        run_calculate_reconstruction_error=MagicMock(return_value=True),
        run_train_prediction_models=MagicMock(return_value=True),
        run_predict_datasets=MagicMock(return_value=True),
        run_calculate_prediction_error=MagicMock(return_value=True),
        load_prediction_models_config=MagicMock(return_value=pred),
    ):
        result = runs.run_pipeline_full(config, pred_config=pred)

    assert not result.ok
    assert result.failed_step == "clean_datasets"
    assert result.steps_completed == []

"""Tests for prediction_metrics registry (built-ins + custom registration)."""

from __future__ import annotations

import numpy as np
import pytest

from prediction_metrics._registry import (
    _BUILTIN_COMPUTE,
    _PRIMARY_ORDER,
    _SPECS,
    _USER_METRICS,
    PredictionMetricSpec,
    compute_prediction_metrics,
    get_metric_spec,
    infer_lower_is_better,
    list_primary_metric_keys,
    register_prediction_metric,
)


@pytest.fixture(autouse=True)
def _clean_user_metrics():
    saved_order = list(_PRIMARY_ORDER)
    saved_specs = dict(_SPECS)
    saved_user = dict(_USER_METRICS)
    yield
    _USER_METRICS.clear()
    _USER_METRICS.update(saved_user)
    _SPECS.clear()
    _SPECS.update(saved_specs)
    _PRIMARY_ORDER.clear()
    _PRIMARY_ORDER.extend(saved_order)


BUILT_IN_KEYS = {"mape", "smape", "mase", "mae", "rmse"}


class TestBuiltinMetrics:
    def test_builtins_registered(self):
        keys = set(list_primary_metric_keys())
        assert BUILT_IN_KEYS.issubset(keys)

    def test_builtin_specs_have_correct_type(self):
        for key in BUILT_IN_KEYS:
            spec = get_metric_spec(key)
            assert isinstance(spec, PredictionMetricSpec)

    def test_all_lower_is_better(self):
        for key in BUILT_IN_KEYS:
            assert infer_lower_is_better(key) is True


class TestComputeBuiltins:
    @pytest.fixture()
    def arrays(self):
        return (
            np.array([10.0, 20.0, 30.0]),
            np.array([11.0, 19.0, 28.0]),
            np.array([5.0, 8.0, 12.0, 15.0, 18.0]),
        )

    def test_compute_all_with_train(self, arrays):
        yt, yp, train = arrays
        result = compute_prediction_metrics(yt, yp, train=train)
        for key in BUILT_IN_KEYS:
            assert key in result
            assert np.isfinite(result[key])

    def test_compute_subset(self, arrays):
        yt, yp, _ = arrays
        result = compute_prediction_metrics(yt, yp, metric_keys=["mae", "rmse"])
        assert "mae" in result
        assert "rmse" in result
        assert "mape" not in result

    def test_mase_nan_without_train(self, arrays):
        yt, yp, _ = arrays
        result = compute_prediction_metrics(yt, yp, metric_keys=["mase"])
        assert np.isnan(result["mase"])


class TestCustomRegistration:
    def test_register_and_compute(self):
        def max_err(y_true, y_pred, *, train=None):
            return float(np.max(np.abs(y_true - y_pred)))

        register_prediction_metric(
            "test_max_err", "Test Max Error", lower_is_better=True, compute=max_err
        )
        assert "test_max_err" in list_primary_metric_keys()
        spec = get_metric_spec("test_max_err")
        assert spec.lower_is_better is True

        result = compute_prediction_metrics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.5, 3.0]),
            metric_keys=["test_max_err"],
        )
        assert result["test_max_err"] == pytest.approx(0.5)

    def test_register_higher_is_better(self):
        def r2_like(y_true, y_pred, *, train=None):
            return 0.95

        register_prediction_metric(
            "test_r2_pred", "R2-like", lower_is_better=False, compute=r2_like
        )
        assert infer_lower_is_better("test_r2_pred") is False

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_prediction_metric("", "X", True, lambda a, b, **k: 0.0)

    def test_reserved_key_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            register_prediction_metric("max_error", "X", True, lambda a, b, **k: 0.0)

    def test_builtin_override_raises(self):
        with pytest.raises(ValueError, match="Cannot override"):
            register_prediction_metric("mae", "My MAE", True, lambda a, b, **k: 0.0)

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            get_metric_spec("nonexistent_pred_metric_42")

"""Tests for reconstruction_metrics registry (built-ins + custom registration)."""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction_metrics._registry import (
    _BUILTIN_COMPUTE,
    _PRIMARY_ORDER,
    _SPECS,
    _USER_METRICS,
    ReconstructionMetricSpec,
    compute_reconstruction_metrics,
    get_metric_spec,
    infer_lower_is_better,
    list_primary_metric_keys,
    optimization_loss,
    register_reconstruction_metric,
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


BUILT_IN_KEYS = {"mad", "mae", "rmse", "r2", "smape"}


class TestBuiltinMetrics:
    def test_builtins_registered(self):
        keys = set(list_primary_metric_keys())
        assert BUILT_IN_KEYS.issubset(keys)

    def test_builtin_specs_have_correct_type(self):
        for key in BUILT_IN_KEYS:
            spec = get_metric_spec(key)
            assert isinstance(spec, ReconstructionMetricSpec)

    def test_lower_is_better_flags(self):
        assert infer_lower_is_better("mad") is True
        assert infer_lower_is_better("mae") is True
        assert infer_lower_is_better("rmse") is True
        assert infer_lower_is_better("r2") is False
        assert infer_lower_is_better("smape") is True


class TestComputeBuiltins:
    @pytest.fixture()
    def arrays(self):
        return np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.2, 2.8])

    def test_compute_all_builtins(self, arrays):
        yt, yp = arrays
        result = compute_reconstruction_metrics(yt, yp)
        for key in BUILT_IN_KEYS:
            assert key in result
            assert np.isfinite(result[key])

    def test_compute_subset(self, arrays):
        yt, yp = arrays
        result = compute_reconstruction_metrics(yt, yp, metric_keys=["mad", "rmse"])
        assert "mad" in result
        assert "rmse" in result
        assert "smape" not in result

    def test_auxiliary_fields_present(self, arrays):
        yt, yp = arrays
        result = compute_reconstruction_metrics(yt, yp)
        assert "max_diff" in result
        assert "min_diff" in result
        assert "std_diff" in result
        assert "n_missing" in result


class TestCustomRegistration:
    def test_register_and_compute(self):
        def my_metric(y_true, y_pred):
            return float(np.max(np.abs(y_true - y_pred)))

        register_reconstruction_metric("test_custom", "Test Custom", lower_is_better=True, compute=my_metric)
        assert "test_custom" in list_primary_metric_keys()
        spec = get_metric_spec("test_custom")
        assert spec.lower_is_better is True

        result = compute_reconstruction_metrics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.5, 3.0]),
            metric_keys=["test_custom"],
        )
        assert result["test_custom"] == pytest.approx(0.5)

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_reconstruction_metric("", "X", True, lambda a, b: 0.0)

    def test_reserved_key_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            register_reconstruction_metric("max_diff", "X", True, lambda a, b: 0.0)

    def test_builtin_override_raises(self):
        with pytest.raises(ValueError, match="Cannot override"):
            register_reconstruction_metric("mad", "My MAD", True, lambda a, b: 0.0)

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            get_metric_spec("nonexistent_metric_42")


class TestOptimizationLoss:
    def test_lower_is_better(self):
        assert optimization_loss(3.0, lower_is_better=True) == 3.0

    def test_higher_is_better_negated(self):
        assert optimization_loss(0.9, lower_is_better=False) == -0.9

    def test_nan_returns_inf(self):
        assert optimization_loss(float("nan"), lower_is_better=True) == float("inf")

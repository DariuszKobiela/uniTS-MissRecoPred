"""Tests for framework.plugin_registry (runtime register + entry points)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from framework.plugin_registry import (
    clear_plugin_registry,
    get_reconstruction_models,
    register_prediction_model,
    register_reconstruction_model,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_plugin_registry()
    yield
    clear_plugin_registry()


def test_register_reconstruction_model_appears_in_get():
    def fake_rec(s: pd.Series) -> pd.Series:
        return s

    register_reconstruction_model("plugin_rec_test", fake_rec)
    reg = get_reconstruction_models()
    assert "plugin_rec_test" in reg
    assert reg["plugin_rec_test"] is fake_rec


def test_register_reconstruction_duplicate_without_overwrite_raises():
    def a(s: pd.Series) -> pd.Series:
        return s

    def b(s: pd.Series) -> pd.Series:
        return s * 2

    register_reconstruction_model("dup_rec", a)
    with pytest.raises(ValueError, match="already registered"):
        register_reconstruction_model("dup_rec", b)
    register_reconstruction_model("dup_rec", b, overwrite=True)
    assert get_reconstruction_models()["dup_rec"] is b


def test_register_prediction_model_gpu_deterministic_flags():
    def pred(train, horizon, **kwargs):
        return pd.Series([0.0] * horizon)

    register_prediction_model("plugin_pred_test", pred, gpu=True, deterministic=True)
    from prediction_models import is_deterministic_model, is_gpu_model

    assert is_gpu_model("plugin_pred_test")
    assert is_deterministic_model("plugin_pred_test")


def test_get_available_models_includes_plugin():
    from prediction_models import get_available_models

    def pred(train, horizon, **kwargs):
        return pd.Series([1.0] * horizon)

    register_prediction_model("avail_pred_plugin", pred)
    assert "avail_pred_plugin" in get_available_models()


def test_registered_reconstruction_overrides_builtin_name():
    """Last merge layer wins: runtime register overrides built-in key."""

    def override_linear(s: pd.Series) -> pd.Series:
        return pd.Series([42.0] * len(s), index=s.index)

    register_reconstruction_model("interpolate_linear", override_linear, overwrite=True)
    reg = get_reconstruction_models()
    out = reg["interpolate_linear"](pd.Series([1.0, 2.0]))
    assert (out == 42.0).all()


def test_entry_point_reconstruction_merged():
    def ep_fn(s: pd.Series) -> pd.Series:
        return s.fillna(0)

    class FakeEP:
        name = "ep_recon_demo"

        def load(self):
            return ep_fn

    with patch(
        "framework.plugin_registry._iter_entry_points",
        return_value=(FakeEP(),),
    ):
        clear_plugin_registry()
        reg = get_reconstruction_models()
        assert "ep_recon_demo" in reg
        assert reg["ep_recon_demo"] is ep_fn

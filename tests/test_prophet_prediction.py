import logging

import numpy as np
import pandas as pd
import pytest

import src.prediction_models.prophet as prophet_module


def test_prophet_silences_backend_loggers(monkeypatch):
    cmdstan_logger = logging.getLogger("cmdstanpy")
    prophet_logger = logging.getLogger("prophet")
    prophet_models_logger = logging.getLogger("prophet.models")
    monkeypatch.setattr(cmdstan_logger, "disabled", False)
    monkeypatch.setattr(prophet_logger, "disabled", False)
    monkeypatch.setattr(prophet_models_logger, "disabled", False)

    prophet_module._silence_backend_logging()

    assert cmdstan_logger.disabled is True
    assert prophet_logger.disabled is True
    assert prophet_models_logger.disabled is True


def test_prophet_preserves_sample_positions_when_subsampling(monkeypatch):
    calls = {}

    class FakeProphet:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def fit(self, frame):
            calls["history"] = frame.copy()
            return self

        def add_seasonality(self, **kwargs):
            calls["seasonality"] = kwargs
            return self

        def predict(self, future):
            calls["future"] = future.copy()
            return pd.DataFrame({"yhat": np.arange(len(future), dtype=float)})

    monkeypatch.setattr(prophet_module, "Prophet", FakeProphet)
    series = pd.Series(np.arange(10.0), index=pd.RangeIndex(10))
    forecast = prophet_module.predict_prophet(
        series, horizon=3, max_points=4, seasonal_period=24
    )

    assert len(calls["history"]) == 4
    assert calls["history"]["ds"].iloc[-1] == pd.Timestamp("2000-01-01 09:00:00")
    assert calls["future"]["ds"].iloc[0] == pd.Timestamp("2000-01-01 10:00:00")
    assert calls["seasonality"]["period"] == 1.0
    assert forecast.index.equals(pd.RangeIndex(10, 13))
    assert forecast.attrs["fallback_used"] is False


def test_prophet_reports_trend_fallback(monkeypatch):
    class FailingProphet:
        def __init__(self, **kwargs):
            pass

        def fit(self, frame):
            raise RuntimeError("fit failed")

    monkeypatch.setattr(prophet_module, "Prophet", FailingProphet)
    with pytest.warns(RuntimeWarning, match="trend fallback"):
        forecast = prophet_module.predict_prophet(pd.Series([1.0, 2.0, 3.0]), 2)

    np.testing.assert_allclose(forecast.to_numpy(), [4.0, 5.0])
    assert forecast.attrs["fallback_used"] is True
    assert forecast.attrs["fallback_type"] == "linear_trend"
    assert "fit failed" in forecast.attrs["fallback_reason"]


@pytest.mark.parametrize("horizon", [0, -1])
def test_prophet_rejects_non_positive_horizon(horizon):
    with pytest.raises(ValueError, match="horizon"):
        prophet_module.predict_prophet(pd.Series([1.0, 2.0]), horizon)

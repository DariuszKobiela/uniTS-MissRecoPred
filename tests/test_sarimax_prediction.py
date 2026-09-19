import numpy as np
import pandas as pd
import pytest

import src.prediction_models.sarimax as sarimax_module


class _SuccessfulFit:
    def forecast(self, steps):
        return np.zeros(steps)


def test_sarimax_uses_requested_seasonality_and_level_forecasts(monkeypatch):
    calls = []

    class FakeSarimax:
        def __init__(self, endog, **kwargs):
            calls.append((endog, kwargs))

        def fit(self, **kwargs):
            return _SuccessfulFit()

    monkeypatch.setattr(sarimax_module, "SARIMAX", FakeSarimax)
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    series = pd.Series(np.arange(24.0), index=index)

    forecast = sarimax_module.predict_sarimax(
        series, 3, order=(2, 1, 0), seasonal_order=(1, 1, 1, 7)
    )

    assert calls[0][1]["order"] == (2, 1, 0)
    assert calls[0][1]["seasonal_order"] == (1, 1, 1, 7)
    assert calls[0][1]["simple_differencing"] is False
    assert forecast.index.equals(pd.date_range("2024-01-25", periods=3, freq="D"))
    assert forecast.attrs["fallback_used"] is False


def test_sarimax_reports_simplified_model_fallback(monkeypatch):
    calls = 0

    class FailThenSucceed:
        def __init__(self, endog, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("requested model failed")

        def fit(self, **kwargs):
            return _SuccessfulFit()

    monkeypatch.setattr(sarimax_module, "SARIMAX", FailThenSucceed)
    series = pd.Series(np.arange(20.0))

    with pytest.warns(RuntimeWarning, match="simplified fallback"):
        forecast = sarimax_module.predict_sarimax(
            series, 2, seasonal_order=(1, 1, 1, 12)
        )

    assert forecast.attrs["fallback_used"] is True
    assert forecast.attrs["fallback_type"] == "simplified_sarimax"
    assert "requested model failed" in forecast.attrs["fallback_reason"]
    assert forecast.attrs["fitted_seasonal_order"] == "(0, 0, 0, 0)"


def test_sarimax_reports_trend_fallback(monkeypatch):
    class AlwaysFail:
        def __init__(self, endog, **kwargs):
            raise RuntimeError("fit failed")

    monkeypatch.setattr(sarimax_module, "SARIMAX", AlwaysFail)
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.warns(RuntimeWarning, match="trend fallback"):
        forecast = sarimax_module.predict_sarimax(series, 2)

    np.testing.assert_allclose(forecast.to_numpy(), [6.0, 7.0])
    assert forecast.attrs["fallback_used"] is True
    assert forecast.attrs["fallback_type"] == "linear_trend"
    assert "fit failed" in forecast.attrs["fallback_reason"]

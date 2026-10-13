import numpy as np
import pandas as pd

import src.prediction_models.holt_winters as holt_winters_module


class _SuccessfulFit:
    def forecast(self, horizon):
        return np.zeros(horizon)


def test_holt_winters_uses_dataset_seasonality_and_recent_cycles(monkeypatch):
    calls = []

    class FakeExponentialSmoothing:
        def __init__(self, series, **kwargs):
            calls.append((series.copy(), kwargs))

        def fit(self, **kwargs):
            calls[-1] = (*calls[-1], kwargs)
            return _SuccessfulFit()

    monkeypatch.setattr(holt_winters_module, "ExponentialSmoothing", FakeExponentialSmoothing)
    series = pd.Series(np.arange(100.0), index=pd.RangeIndex(100))

    forecast = holt_winters_module.predict_holt_winters(
        series, horizon=3, seasonal_period=12, max_seasonal_cycles=4
    )

    assert len(calls[0][0]) == 48
    assert calls[0][1]["seasonal"] == "add"
    assert calls[0][1]["seasonal_periods"] == 12
    assert calls[0][2]["optimized"] is True
    assert forecast.index.equals(pd.RangeIndex(100, 103))
    assert forecast.attrs["fallback_used"] is False


def test_holt_winters_reports_nonseasonal_fallback(monkeypatch):
    calls = []

    class FailSeasonal:
        def __init__(self, series, **kwargs):
            calls.append(kwargs)
            if kwargs["seasonal"] is not None:
                raise RuntimeError("seasonal fit failed")

        def fit(self, **kwargs):
            return _SuccessfulFit()

    monkeypatch.setattr(holt_winters_module, "ExponentialSmoothing", FailSeasonal)
    forecast = holt_winters_module.predict_holt_winters(
        pd.Series(np.arange(30.0)), horizon=2, seasonal_period=5
    )

    assert len(calls) == 2
    assert calls[1]["seasonal"] is None
    assert forecast.attrs["fallback_used"] is True
    assert forecast.attrs["fallback_type"] == "nonseasonal_holt"
    assert "seasonal fit failed" in forecast.attrs["fallback_reason"]

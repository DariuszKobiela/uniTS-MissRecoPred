import numpy as np
import pandas as pd
import pytest

from utils.rolling_origins import (
    create_lag_features,
    create_lag_features_with_missing,
    cumulative_horizon_metrics,
    evaluate_forecast_slices,
    expanding_history,
    fit_predict_xgboost_direct_missing,
    fit_predict_xgboost_local,
    lead_time_bin_metrics,
    plan_rolling_origins,
    predict_persistence,
    predict_seasonal_naive,
)


def test_five_origins_are_spread_across_available_test_range():
    origins = plan_rolling_origins(test_length=720, horizon=180, n_origins=5)

    assert len(origins) == 5
    assert origins[0].test_offset == 0
    assert origins[-1].test_stop == 720
    assert [origin.test_offset for origin in origins] == sorted({origin.test_offset for origin in origins})


def test_full_test_horizon_has_only_one_valid_origin():
    origins = plan_rolling_origins(test_length=720, horizon=720, n_origins=5)

    assert len(origins) == 1
    assert origins[0].test_offset == 0
    assert origins[0].test_stop == 720


def test_expanding_history_reveals_only_prefix_before_origin():
    train = pd.Series([1.0, 2.0, 3.0])
    test = pd.Series([10.0, 20.0, 30.0, 40.0])

    history = expanding_history(train, test, test_offset=2)

    assert history.tolist() == [1.0, 2.0, 3.0, 10.0, 20.0]
    assert 30.0 not in history.tolist()
    assert 40.0 not in history.tolist()


@pytest.mark.parametrize("bad_horizon", [0, 11])
def test_origin_plan_rejects_invalid_horizon(bad_horizon):
    with pytest.raises(ValueError, match="horizon"):
        plan_rolling_origins(test_length=10, horizon=bad_horizon, n_origins=5)


def test_lag_features_do_not_cross_target_position():
    features, targets = create_lag_features(np.arange(6.0), lag=3)

    np.testing.assert_array_equal(features, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(targets, [3, 4, 5])


def test_local_xgboost_refit_returns_requested_horizon():
    history = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 60)))
    predictions = fit_predict_xgboost_local(
        history,
        horizon=4,
        params={
            "lags": 5,
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
        },
        seed=7,
    )

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_persistence_repeats_last_value():
    history = pd.Series([1.0, 2.0, 3.5])
    forecast = predict_persistence(history, horizon=4)
    np.testing.assert_array_equal(forecast, [3.5, 3.5, 3.5, 3.5])


def test_seasonal_naive_uses_lag_period():
    history = pd.Series([1.0, 2.0, 3.0, 4.0])
    forecast = predict_seasonal_naive(history, horizon=3, seasonal_period=2)
    np.testing.assert_array_equal(forecast, [3.0, 4.0, 3.0])


def test_direct_missing_xgboost_handles_nans():
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    history = pd.Series(values)
    forecast = fit_predict_xgboost_direct_missing(
        history,
        horizon=3,
        params={"lags": 4, "n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "n_jobs": 1},
        seed=1,
    )
    assert forecast.shape == (3,)
    assert np.isfinite(forecast).all()


def test_create_lag_features_with_missing_includes_mask():
    values = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    features, targets = create_lag_features_with_missing(values, lag=3)
    assert features.shape[1] == 6
    assert len(targets) >= 1


def test_hmax_forecast_slices_into_cumulative_and_lead_bins():
    actual = np.arange(1.0, 13.0)
    predicted = actual + 0.5
    horizons = [3, 6, 12]
    rows = evaluate_forecast_slices(
        actual,
        predicted,
        horizons,
        train_history=actual[:20],
        metric_keys=["mae"],
        include_lead_time_bins=True,
    )
    cumulative = [row for row in rows if row["metric_scope"] == "cumulative"]
    bins = [row for row in rows if row["metric_scope"] == "lead_time_bin"]
    assert {row["forecast_horizon"] for row in cumulative} == {3, 6, 12}
    assert len(bins) == 3
    assert cumulative[-1]["n_samples"] == 12
    assert bins[0]["lead_time_start"] == 1
    assert bins[-1]["lead_time_end"] == 12


def test_three_origins_for_vibration_horizon():
    origins = plan_rolling_origins(test_length=24, horizon=24, n_origins=3)
    assert len(origins) == 1
    origins = plan_rolling_origins(test_length=48, horizon=24, n_origins=3)
    assert len(origins) == 3

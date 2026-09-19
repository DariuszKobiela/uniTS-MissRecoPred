import numpy as np
import pandas as pd
import pytest

from utils.rolling_origins import (
    create_lag_features,
    expanding_history,
    fit_predict_xgboost_local,
    plan_rolling_origins,
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

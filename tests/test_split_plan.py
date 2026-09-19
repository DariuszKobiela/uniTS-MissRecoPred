"""Tests for three-way split planning."""

from __future__ import annotations

import pytest

from utils.horizon_recommender import HorizonConstraints
from utils.split_plan import (
    compute_validation_length,
    largest_safe_rolling_test_length,
    min_rolling_test_length,
    plan_three_way_split,
    resolve_n_origins,
)


def test_min_rolling_test_length_for_five_origins():
    assert min_rolling_test_length(h_max=720, n_origins=5) == 724


def test_largest_safe_holdout_uses_twenty_percent_when_possible():
    length, _ = largest_safe_rolling_test_length(
        series_length=8000,
        h_max=1440,
        n_origins=5,
        constraints=HorizonConstraints(max_holdout_share=0.20, min_train_length=200),
        min_validation_length=64,
    )
    assert length == 1600


def test_three_way_split_is_disjoint():
    boundaries = plan_three_way_split(
        series_length=8000,
        h_max=1440,
        n_origins=5,
        constraints=HorizonConstraints(max_holdout_share=0.20, min_train_length=200),
        validation_share=0.10,
        validation_min_samples=64,
        validation_max_samples=500,
    )
    assert boundaries.reconstruction_end < boundaries.validation_end <= boundaries.test_start
    assert boundaries.test_length >= 1440
    assert boundaries.reconstruction_length >= 200


def test_resolve_n_origins_per_dataset():
    mapping = {
        "boiler_outlet_temp_univ.csv": 5,
        "vibration_sensor_S1.csv": 3,
    }
    assert resolve_n_origins("boiler_outlet_temp_univ.csv", mapping) == 5
    assert resolve_n_origins("vibration_sensor_S1.csv", mapping) == 3


def test_validation_length_respects_bounds():
    assert compute_validation_length(1000, share=0.10, min_samples=64, max_samples=500) == 100


def test_short_series_raises():
    with pytest.raises(ValueError, match="cannot host"):
        plan_three_way_split(
            series_length=300,
            h_max=1440,
            n_origins=5,
        )

"""Tests for forecast horizon recommendation utilities."""

from __future__ import annotations

import pandas as pd

from utils.horizon_recommender import (
    HorizonConstraints,
    STATUS_IDEAL,
    STATUS_MAXIMUM_SAFE,
    STATUS_SAFE,
    STATUS_UNSAFE,
    analyze_profiles_batch,
    format_duration_aggregated,
    format_horizon_span,
    infer_series_profile,
    load_h_long_lookup,
    metadata_document,
    recommend_horizons_for_profile,
)


def _index(n: int, freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq=freq)


def test_hourly_regular_interval_not_named_group():
    profile = infer_series_profile("hourly.csv", _index(1000))
    assert profile.sampling_interval_seconds == 3600
    assert profile.freq_confidence > 0.9
    assert "hourly" not in profile.sampling_label.lower() or "every" in profile.sampling_label

    rec = recommend_horizons_for_profile(profile)
    assert rec.h_short == 12
    assert rec.h_long == 96
    assert rec.status in {STATUS_IDEAL, STATUS_SAFE}
    assert rec.holdout_share <= 0.20
    assert rec.train_length >= 200


def test_daily_n212_uses_short_series_pair():
    profile = infer_series_profile("daily.csv", _index(212, freq="D"))
    rec = recommend_horizons_for_profile(profile)
    assert rec.h_short == 12
    assert rec.h_long == 24
    assert rec.h_long / rec.n <= 0.20
    assert rec.status == STATUS_MAXIMUM_SAFE


def test_weekly_uses_short_series_pair_when_96_does_not_fit():
    profile = infer_series_profile("weekly.csv", _index(221, freq="W"))
    rec = recommend_horizons_for_profile(profile)
    assert rec.h_short == 12
    assert rec.h_long == 24
    assert rec.h_long / rec.n <= 0.20
    assert rec.status == STATUS_MAXIMUM_SAFE


def test_short_series_uses_12_and_24():
    profile = infer_series_profile("daily_short.csv", _index(211, freq="D"))
    rec = recommend_horizons_for_profile(profile)
    assert rec.h_short == 12
    assert rec.h_long == 24
    assert rec.h_long <= int(0.20 * 211)


def test_irregular_uses_mean_of_common_gap_not_15min_bucket():
    timestamps = pd.to_datetime(
        [
            "2024-04-03 01:30:00",
            "2024-04-03 01:55:00",
            "2024-04-03 02:00:00",
            "2024-04-03 02:45:00",
            "2024-04-03 03:10:00",
            "2024-04-03 03:25:00",
        ]
    )
    profile = infer_series_profile("vibration_sensor_S1.csv", timestamps)
    assert profile.is_irregular
    assert profile.sampling_interval_seconds is not None
    assert "15min" not in profile.sampling_label
    rec = recommend_horizons_for_profile(profile)
    assert rec.h_long >= 1
    assert rec.h_long < rec.n


def test_numeric_index_constraint_fallback():
    index = pd.Index(range(250))
    profile = infer_series_profile("numeric.csv", index)
    assert profile.sampling_interval_seconds is None

    recs = analyze_profiles_batch([profile])
    rec = recs[0]
    assert rec.h_long <= int(0.20 * 250)
    assert rec.train_length == rec.n - rec.h_long
    assert rec.status != STATUS_UNSAFE


def test_does_not_share_static_horizons_across_series():
    profiles = [
        infer_series_profile("d1.csv", _index(500, freq="D")),
        infer_series_profile("d2.csv", _index(212, freq="D")),
    ]
    recs = analyze_profiles_batch(profiles)
    by_id = {r.series_id: r for r in recs}
    assert by_id["d1.csv"].h_long == 96
    assert by_id["d2.csv"].h_long == 24
    assert by_id["d2.csv"].status != STATUS_UNSAFE


def test_metadata_document_and_lookup():
    profiles = [infer_series_profile("d1.csv", _index(300, freq="D"))]
    recs = analyze_profiles_batch(profiles)
    doc = metadata_document(
        recs,
        input_dir="data/1_cleaned_data",
        output_dir="data/1_5_horizon_recommendation",
        constraints=HorizonConstraints(),
        generated_at="2026-01-01T00:00:00+00:00",
    )
    lookup = load_h_long_lookup(doc)
    assert lookup["d1.csv"] == recs[0].h_long
    assert "sampling_label" in doc["series"]["d1.csv"]
    assert recs[0].status != STATUS_UNSAFE


def test_vibration_s1_length_uses_12_and_24():
    profile = infer_series_profile("tight.csv", _index(210, freq="D"))
    rec = recommend_horizons_for_profile(profile)
    assert rec.h_short == 12
    assert rec.h_long == 24
    assert rec.train_length == 186
    assert rec.status == STATUS_MAXIMUM_SAFE


def test_tiny_series_still_prefers_12_and_24_if_holdout_ok():
    profile = infer_series_profile("tiny.csv", _index(150, freq="D"))
    rec = recommend_horizons_for_profile(profile)
    assert rec.h_short == 12
    assert rec.h_long == 24
    assert rec.h_long / rec.n <= 0.20


def test_duration_uses_most_aggregated_unit():
    assert format_duration_aggregated(5) == "5 s"
    assert format_duration_aggregated(60) == "1 min"
    assert format_duration_aggregated(3600) == "1 h"
    assert format_duration_aggregated(3 * 3600) == "3 h"
    assert format_duration_aggregated(2 * 86400) == "2 d"
    assert format_duration_aggregated(14 * 86400) == "2 weeks"
    assert format_duration_aggregated(365.25 * 86400) == "1 year"


def test_experiment_horizons_use_longest_for_split():
    mapping = {
        "boiler_outlet_temp_univ.csv": [12, 96, 180, 720, 1440],
        "pump_sensor_28_univ.csv": [12, 96, 360, 1440],
        "vibration_sensor_S1.csv": [12, 24],
    }
    boiler = infer_series_profile(
        "boiler_outlet_temp_univ.csv", pd.date_range("2020-01-01", periods=5000, freq="5s")
    )
    rec = recommend_horizons_for_profile(boiler, experiment_horizons=mapping)
    assert rec.horizons == [12, 96, 180, 720, 1440]
    assert rec.h_long == 1440
    assert rec.train_length == 5000 - 1440

    lookup = load_h_long_lookup(
        metadata_document(
            [rec],
            input_dir="in",
            output_dir="out",
            constraints=HorizonConstraints(),
            generated_at="2026-01-01T00:00:00+00:00",
        )
    )
    assert lookup["boiler_outlet_temp_univ.csv"] == 1440
    assert format_horizon_span(720, 5.0) == "1 h"
    assert format_horizon_span(2160, 5.0) == "3 h"
    assert format_horizon_span(60, 60.0) == "1 h"
    assert format_horizon_span(10, None) == "—"


def test_subhourly_long_series_use_literature_12_and_96():
    five_s = infer_series_profile(
        "five_s.csv", pd.date_range("2020-01-01", periods=2000, freq="5s")
    )
    rec_5s = recommend_horizons_for_profile(five_s)
    assert rec_5s.h_short == 12
    assert rec_5s.h_long == 96
    assert format_horizon_span(rec_5s.h_short, five_s.sampling_interval_seconds) == "1 min"
    assert format_horizon_span(rec_5s.h_long, five_s.sampling_interval_seconds) == "8 min"

    one_min = infer_series_profile(
        "one_min.csv", pd.date_range("2020-01-01", periods=2000, freq="min")
    )
    rec_min = recommend_horizons_for_profile(one_min)
    assert rec_min.h_short == 12
    assert rec_min.h_long == 96
    assert format_horizon_span(rec_min.h_short, one_min.sampling_interval_seconds) == "12 min"
    assert format_horizon_span(rec_min.h_long, one_min.sampling_interval_seconds) == "1.6 h"

"""Behavioural tests for simulated MCAR, MAR, and MNAR mechanisms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingness_techniques.mar import apply_mar
from missingness_techniques.mcar import apply_mcar
from missingness_techniques.mnar import apply_mnar


@pytest.mark.parametrize("mechanism", [apply_mcar, apply_mar, apply_mnar])
def test_mechanisms_are_reproducible_and_preserve_input(mechanism):
    source = pd.Series(np.linspace(-2.0, 3.0, 1000))
    original = source.copy()

    first = mechanism(source, 0.2, seed=123)
    second = mechanism(source, 0.2, seed=123)

    pd.testing.assert_series_equal(source, original)
    assert first.isna().sum() == 200
    assert first.isna().equals(second.isna())


@pytest.mark.parametrize("mechanism", [apply_mcar, apply_mar, apply_mnar])
@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_mechanisms_reject_invalid_rates(mechanism, rate):
    with pytest.raises(ValueError, match="between 0 and 1"):
        mechanism(pd.Series([1.0, 2.0, 3.0]), rate, seed=1)


def test_mar_mask_depends_on_time_not_series_values():
    increasing = pd.Series(np.arange(5000, dtype=float))
    reversed_values = increasing.iloc[::-1].reset_index(drop=True)

    result_a = apply_mar(increasing, 0.2, seed=7)
    result_b = apply_mar(reversed_values, 0.2, seed=7)
    missing_positions = np.flatnonzero(result_a.isna().to_numpy())

    assert result_a.isna().equals(result_b.isna())
    assert missing_positions.mean() > (len(increasing) - 1) / 2


def test_mnar_prefers_values_far_from_the_median():
    values = np.zeros(5000, dtype=float)
    values[::10] = 10.0
    source = pd.Series(values)

    result = apply_mnar(source, 0.1, seed=19)
    masked_values = source[result.isna()]

    # Extreme values form only 10% of the source, but value-dependent sampling
    # should make them a clear majority of the removed observations.
    assert (masked_values == 10.0).mean() > 0.5


def test_existing_missing_values_count_towards_target_rate():
    source = pd.Series(np.arange(100, dtype=float))
    source.iloc[:5] = np.nan

    result = apply_mar(source, 0.1, seed=5)

    assert result.isna().sum() == 10
    assert result.iloc[:5].isna().all()

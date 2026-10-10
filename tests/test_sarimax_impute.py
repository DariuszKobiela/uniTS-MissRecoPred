import numpy as np
import pandas as pd
import pytest

from reconstruction_models.sarimax import (
    _imputed_values_are_plausible,
    _starting_parameters,
    sarimax_impute,
)


class _ExplosiveStart:
    start_params = np.array([-2.5, 2.5, 10.0])

    def loglike(self, params):
        params = np.asarray(params, dtype=float)
        if abs(params[0]) > 1:
            return np.nan
        return -10.0


def test_starting_parameters_replace_explosive_default():
    with pytest.warns(RuntimeWarning, match="non-finite likelihood"):
        start = _starting_parameters(_ExplosiveStart(), np.array([1.0, 2.0, 4.0, 7.0]))

    assert start[0] == 0.0
    assert start[1] == 0.0
    assert start[-1] > 0
    assert np.isfinite(_ExplosiveStart().loglike(start))


def test_starting_parameters_keep_a_finite_default():
    model = _ExplosiveStart()
    model.start_params = np.array([0.2, -0.1, 3.0])

    start = _starting_parameters(model, np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(start, model.start_params)


def test_sarimax_impute_fills_an_interior_gap():
    values = np.linspace(0.0, 5.0, 80)
    values[30:45] = np.nan
    series = pd.Series(values)

    imputed = sarimax_impute(series)

    assert np.isfinite(imputed.to_numpy(dtype=float)).all()
    np.testing.assert_array_equal(imputed.iloc[:30].to_numpy(), series.iloc[:30].to_numpy())
    np.testing.assert_array_equal(imputed.iloc[45:].to_numpy(), series.iloc[45:].to_numpy())


def test_plausibility_check_rejects_finite_numerical_explosion():
    observed = np.array([20.0, 30.0, 40.0, 50.0])

    assert _imputed_values_are_plausible(observed, np.array([35.0, 45.0]))
    assert not _imputed_values_are_plausible(observed, np.array([3.43e54]))
    assert not _imputed_values_are_plausible(observed, np.array([2.12e138]))


def test_nonconverged_fit_uses_linear_fallback(monkeypatch):
    import statsmodels.tsa.statespace.sarimax as sarimax_module

    class FakeResults:
        mle_retvals = {"converged": False}

    class FakeModel:
        start_params = np.array([0.0, 0.0, 1.0])

        def __init__(self, *args, **kwargs):
            pass

        def loglike(self, params):
            return -1.0

        def fit(self, **kwargs):
            return FakeResults()

    monkeypatch.setattr(sarimax_module, "SARIMAX", FakeModel)
    series = pd.Series([1.0, np.nan, 3.0, 4.0])

    with pytest.warns(RuntimeWarning, match="optimizer did not converge"):
        result = sarimax_impute(series)

    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0, 4.0])

import numpy as np
import pandas as pd

from reconstruction_models.edge_fallback import (
    EDGE_FALLBACK_MODELS,
    apply_nearest_edge_fallback,
)


def test_edge_fallback_fills_only_unresolved_ends():
    series = pd.Series([np.nan, np.nan, 10.0, 12.0, np.nan])

    filled, count = apply_nearest_edge_fallback(series)

    assert count == 3
    assert filled.tolist() == [10.0, 10.0, 10.0, 12.0, 12.0]


def test_edge_fallback_leaves_complete_series_unchanged():
    series = pd.Series([1.0, 2.0, 3.0])

    filled, count = apply_nearest_edge_fallback(series)

    assert count == 0
    assert filled.tolist() == [1.0, 2.0, 3.0]


def test_edge_fallback_models_are_the_endpoint_methods_only():
    assert EDGE_FALLBACK_MODELS == {
        "impute_ffill",
        "impute_bfill",
        "interpolate_linear",
        "interpolate_cubic",
        "interpolate_pchip",
    }

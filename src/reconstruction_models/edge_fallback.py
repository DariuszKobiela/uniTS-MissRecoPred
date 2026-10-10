"""Nearest-observation fill for gaps a method cannot extrapolate at the ends."""

from __future__ import annotations

import pandas as pd

EDGE_FALLBACK_MODELS = frozenset(
    {
        "impute_ffill",
        "impute_bfill",
        "interpolate_linear",
        "interpolate_cubic",
        "interpolate_pchip",
    }
)


def apply_nearest_edge_fallback(series: pd.Series) -> tuple[pd.Series, int]:
    """Fill NaNs left at the series ends with the nearest finite observation.

    ``ffill`` covers a trailing gap. ``bfill`` then covers a leading gap.
    Interior values already produced by the reconstruction method are unchanged.
    Returns the completed series and the number of samples filled by this step.
    """
    values = pd.Series(series)
    remaining = int(values.isna().sum())
    if remaining == 0:
        return values, 0
    return values.ffill().bfill(), remaining

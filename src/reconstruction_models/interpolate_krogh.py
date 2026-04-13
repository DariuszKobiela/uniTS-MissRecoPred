"""
Krogh Interpolation
Krogh polynomial interpolation through observed data points.
"""

import pandas as pd


def interpolate_krogh(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using Krogh polynomial interpolation.

    Args:
        data: Pandas Series with missing values (NaN)

    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='krogh')

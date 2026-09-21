"""
Spline Interpolation
Uses 2nd order spline interpolation.
"""

import pandas as pd


def interpolate_spline(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using spline interpolation (order=2).
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='spline', order=2)


"""
PCHIP Interpolation
Piecewise Cubic Hermite Interpolating Polynomial - preserves monotonicity.
"""

import pandas as pd


def interpolate_pchip(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using PCHIP (Piecewise Cubic Hermite) interpolation.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='pchip')


"""
Akima Interpolation
Akima interpolation - smooth curve fitting.
"""

import pandas as pd


def interpolate_akima(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using Akima interpolation.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='akima')


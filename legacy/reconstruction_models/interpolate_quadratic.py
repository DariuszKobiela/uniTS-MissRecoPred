"""
Quadratic Interpolation
Uses quadratic (2nd order polynomial) interpolation.
"""

import pandas as pd


def interpolate_quadratic(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using quadratic interpolation.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='quadratic')


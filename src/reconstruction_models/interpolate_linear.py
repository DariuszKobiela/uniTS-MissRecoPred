"""
Linear Interpolation
Uses linear interpolation between valid values.
"""

import pandas as pd


def interpolate_linear(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using linear interpolation.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='linear')


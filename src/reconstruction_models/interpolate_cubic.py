"""
Cubic Interpolation
Uses cubic (3rd order polynomial) interpolation.
"""

import pandas as pd


def interpolate_cubic(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using cubic interpolation.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='cubic')


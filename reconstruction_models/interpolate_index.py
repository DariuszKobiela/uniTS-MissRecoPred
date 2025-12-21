"""
Index-based Interpolation
Uses index values for interpolation weights.
"""

import pandas as pd


def interpolate_index(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using index-based interpolation.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='index')


"""
Nearest Neighbor Interpolation
Uses nearest valid value for interpolation.
"""

import pandas as pd


def interpolate_nearest(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using nearest neighbor method.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='nearest')


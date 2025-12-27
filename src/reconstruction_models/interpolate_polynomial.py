"""
Polynomial Interpolation
Uses 2nd order polynomial interpolation.
"""

import pandas as pd


def interpolate_polynomial(data: pd.Series) -> pd.Series:
    """
    Interpolate missing values using polynomial interpolation (order=2).
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values interpolated
    """
    return data.interpolate(method='polynomial', order=2)


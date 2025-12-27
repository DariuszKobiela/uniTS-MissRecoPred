"""
Median Imputation
Replaces missing values with the median of observed values.
"""

import pandas as pd


def impute_median(data: pd.Series) -> pd.Series:
    """
    Impute missing values using the median of observed values.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values filled with median
    """
    return data.fillna(data.median())


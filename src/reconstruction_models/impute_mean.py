"""
Mean Imputation
Replaces missing values with the mean of observed values.
"""

import pandas as pd


def impute_mean(data: pd.Series) -> pd.Series:
    """
    Impute missing values using the mean of observed values.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values filled with mean
    """
    return data.fillna(data.mean())


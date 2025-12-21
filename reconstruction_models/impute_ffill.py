"""
Forward Fill Imputation
Propagates last valid observation forward to fill missing values.
"""

import pandas as pd


def impute_ffill(data: pd.Series) -> pd.Series:
    """
    Impute missing values using forward fill (propagate last valid observation).
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values forward filled
    """
    return data.fillna(method='ffill')


"""
Backward Fill Imputation
Propagates next valid observation backward to fill missing values.
"""

import pandas as pd


def impute_bfill(data: pd.Series) -> pd.Series:
    """
    Impute missing values using backward fill (propagate next valid observation).
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values backward filled
    """
    return data.fillna(method='bfill')


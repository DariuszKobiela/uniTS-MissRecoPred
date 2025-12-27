"""
Mode Imputation
Replaces missing values with the mode (most frequent value) of observed values.
"""

import pandas as pd


def impute_mode(data: pd.Series) -> pd.Series:
    """
    Impute missing values using the mode of observed values.
    Falls back to mean if no mode exists.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values filled with mode
    """
    mode_vals = data.mode()
    mode_val = mode_vals.iloc[0] if not mode_vals.empty else data.mean()
    return data.fillna(mode_val)


"""
SARIMAX Imputation
Uses SARIMA model to impute missing values based on time series patterns.
Uses Kalman smoothing to estimate missing values.
"""

import pandas as pd
import numpy as np


def sarimax_impute(data: pd.Series) -> pd.Series:
    """
    Impute missing values using SARIMAX model with Kalman smoothing.
    
    Args:
        data: Pandas Series with missing values (NaN)
        
    Returns:
        Pandas Series with missing values imputed
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    missing_mask = data.isna()
    
    if not missing_mask.any():
        return data.copy()
    
    # Check if we have enough non-missing values
    non_missing_count = (~missing_mask).sum()
    if non_missing_count < 3:
        raise ValueError(f"Not enough non-missing values for SARIMAX model (have {non_missing_count}, need at least 3)")
    
    # Create a copy with NaN values for SARIMAX
    # SARIMAX can handle missing values in the endogenous variable
    data_with_nan = data.copy()
    
    # Fit SARIMAX model - it will use Kalman filter to handle missing values
    model = SARIMAX(data_with_nan, order=(1, 1, 1))
    fitted = model.fit(disp=False)
    
    # Use smoothed states to get estimates for missing values
    # The smoothed states provide optimal estimates given all data
    smoothed = fitted.smoother_results.smoothed_state[0]
    
    # Create result series
    result = data.copy()
    
    # Fill missing values with smoothed estimates
    result[missing_mask] = smoothed[missing_mask]
    
    return result


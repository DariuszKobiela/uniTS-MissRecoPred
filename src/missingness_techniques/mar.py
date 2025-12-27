"""
Missing At Random (MAR)
Missing values depend on observed data but not on the missing values themselves.
"""

import pandas as pd
import numpy as np


def apply_mar(data: pd.Series, missing_rate: float, seed: int = None) -> pd.Series:
    """
    Introduce Missing At Random (MAR) pattern.
    
    In MAR, the probability of being missing depends on other observed variables
    or characteristics of the data. Here, we make values more likely to be missing
    when they deviate more from the median (higher variability = higher missing probability).
    
    Args:
        data: Original Pandas Series
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series with MAR missing values (NaN)
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_copy = data.copy()
    n_missing = int(len(data) * missing_rate)
    
    # Calculate deviation from median as basis for missing probability
    data_for_probs = data.fillna(data.median())
    diff_from_median = np.abs(data_for_probs.values - data_for_probs.median())
    
    # Handle edge cases
    if diff_from_median.sum() == 0 or np.isnan(diff_from_median.sum()) or np.isinf(diff_from_median.sum()):
        # If all values are the same, fall back to MCAR
        print("Warning: All values identical, falling back to MCAR")
        missing_indices = np.random.choice(len(data), n_missing, replace=False)
    else:
        # Create probability distribution: higher deviation = higher missing probability
        probs = diff_from_median / diff_from_median.sum()
        
        # Ensure no NaN or inf in probabilities
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            print("Warning: Invalid probabilities, falling back to MCAR")
            missing_indices = np.random.choice(len(data), n_missing, replace=False)
        else:
            missing_indices = np.random.choice(len(data), n_missing, replace=False, p=probs)
    
    # Set selected indices to NaN
    data_copy.iloc[missing_indices] = np.nan
    
    print(f"✓ Applied MAR: {n_missing}/{len(data)} values ({missing_rate*100:.1f}%) set to missing")
    print(f"  (Missing probability based on deviation from median)")
    
    return data_copy


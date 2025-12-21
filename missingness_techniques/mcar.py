"""
Missing Completely At Random (MCAR)
Missing values are randomly distributed with equal probability across all observations.
"""

import pandas as pd
import numpy as np


def apply_mcar(data: pd.Series, missing_rate: float, seed: int = None) -> pd.Series:
    """
    Introduce Missing Completely At Random (MCAR) pattern.
    
    In MCAR, the probability of being missing is the same for all observations,
    regardless of the values of the time series.
    
    Args:
        data: Original Pandas Series
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series with MCAR missing values (NaN)
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_copy = data.copy()
    n_missing = int(len(data) * missing_rate)
    
    # Randomly select indices to make missing
    missing_indices = np.random.choice(len(data), n_missing, replace=False)
    
    # Set selected indices to NaN
    data_copy.iloc[missing_indices] = np.nan
    
    print(f"✓ Applied MCAR: {n_missing}/{len(data)} values ({missing_rate*100:.1f}%) set to missing")
    
    return data_copy


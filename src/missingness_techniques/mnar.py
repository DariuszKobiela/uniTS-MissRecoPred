"""
Missing Not At Random (MNAR)
Missing values depend on the unobserved values themselves - systematic pattern.
"""

import pandas as pd
import numpy as np


def apply_mnar(data: pd.Series, missing_rate: float, seed: int = None) -> pd.Series:
    """
    Introduce Missing Not At Random (MNAR) pattern.
    
    In MNAR, the probability of being missing depends on the value itself or
    systematic patterns. Here, we make later values in the series more likely
    to be missing (simulating sensor degradation or dropout over time).
    
    Args:
        data: Original Pandas Series
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series with MNAR missing values (NaN)
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_copy = data.copy()
    n_missing = int(len(data) * missing_rate)
    
    # Create linearly increasing weights (later values more likely to be missing)
    weights = np.linspace(0.1, 1.0, len(data))
    probs = weights / weights.sum()
    
    # Select indices based on probability
    missing_indices = np.random.choice(len(data), n_missing, replace=False, p=probs)
    
    # Set selected indices to NaN
    data_copy.iloc[missing_indices] = np.nan
    
    print(f"✓ Applied MNAR: {n_missing}/{len(data)} values ({missing_rate*100:.1f}%) set to missing")
    print(f"  (Missing probability increases over time)")
    
    return data_copy


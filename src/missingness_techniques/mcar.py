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
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be between 0 and 1, got {missing_rate}")

    data_copy = data.copy()
    target_missing = int(len(data) * missing_rate)
    already_missing = int(data_copy.isna().sum())
    n_to_add = max(0, target_missing - already_missing)

    available_indices = np.flatnonzero(data_copy.notna().to_numpy())
    n_to_add = min(n_to_add, len(available_indices))

    # Use a local generator so parallel tasks do not modify NumPy's global RNG.
    rng = np.random.default_rng(seed)
    missing_indices = rng.choice(available_indices, n_to_add, replace=False)

    # Set selected indices to NaN
    data_copy.iloc[missing_indices] = np.nan

    print(
        f"✓ Applied MCAR: {int(data_copy.isna().sum())}/{len(data)} values "
        f"({missing_rate*100:.1f}% target) missing"
    )

    return data_copy


"""Missing At Random (MAR) simulation for a univariate time series.

The missingness probability depends on the fully observed time position, not on
the value that is removed. This represents time-dependent sensor availability.
"""

import pandas as pd
import numpy as np


def apply_mar(data: pd.Series, missing_rate: float, seed: int = None) -> pd.Series:
    """
    Introduce Missing At Random (MAR) pattern.
    
    A normalized time position is treated as a fully observed covariate. Sampling
    weights increase linearly from 0.1 at the beginning of the series to 1.0 at
    the end. Values at the selected positions are never used to construct the
    probabilities, so the mechanism is MAR conditional on time.
    
    Args:
        data: Original Pandas Series
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series with MAR missing values (NaN)
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be between 0 and 1, got {missing_rate}")

    data_copy = data.copy()
    target_missing = int(len(data) * missing_rate)
    already_missing = int(data_copy.isna().sum())
    n_to_add = max(0, target_missing - already_missing)

    available_indices = np.flatnonzero(data_copy.notna().to_numpy())
    n_to_add = min(n_to_add, len(available_indices))

    # Time/position remains observed after the corresponding value vanishes.
    time_weights = np.linspace(0.1, 1.0, len(data), dtype=np.float64)
    available_weights = time_weights[available_indices]
    probabilities = available_weights / available_weights.sum()

    rng = np.random.default_rng(seed)
    missing_indices = rng.choice(
        available_indices, n_to_add, replace=False, p=probabilities
    )

    # Set selected indices to NaN
    data_copy.iloc[missing_indices] = np.nan

    print(
        f"✓ Applied MAR: {int(data_copy.isna().sum())}/{len(data)} values "
        f"({missing_rate*100:.1f}% target) missing"
    )
    print("  (Missing probability depends on the fully observed time position)")

    return data_copy


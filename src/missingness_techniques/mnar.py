"""Missing Not At Random (MNAR) simulation for a univariate time series.

The probability of removal depends on the value that is subsequently hidden.
Extreme values are sampled more often than values close to the median.
"""

import pandas as pd
import numpy as np


def apply_mnar(data: pd.Series, missing_rate: float, seed: int = None) -> pd.Series:
    """
    Introduce Missing Not At Random (MNAR) pattern.
    
    The probability of being missing depends on the value itself. Sampling
    weights are proportional to the absolute deviation from the median, with a
    small positive floor so every observed value remains eligible. Access to the
    complete values is intentional here: this function simulates MNAR data and
    then hides the selected values.
    
    Args:
        data: Original Pandas Series
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Pandas Series with MNAR missing values (NaN)
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be between 0 and 1, got {missing_rate}")

    data_copy = data.copy()
    target_missing = int(len(data) * missing_rate)
    already_missing = int(data_copy.isna().sum())
    n_to_add = max(0, target_missing - already_missing)

    available_indices = np.flatnonzero(data_copy.notna().to_numpy())
    n_to_add = min(n_to_add, len(available_indices))

    observed_values = pd.to_numeric(
        data_copy.iloc[available_indices], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(observed_values).all():
        raise ValueError("MNAR requires finite numeric values at observed positions")

    if len(observed_values) == 0:
        missing_indices = np.array([], dtype=int)
    else:
        median = float(np.median(observed_values))
        deviation = np.abs(observed_values - median)
        max_deviation = float(np.max(deviation))

        if max_deviation > 0.0:
            # Range 0.05..1.05: extremes are much more likely, while the
            # positive floor permits any requested rate up to 100%.
            weights = 0.05 + deviation / max_deviation
        else:
            # Value-dependent missingness is not identifiable for a constant
            # series, so uniform sampling is the only meaningful fallback.
            weights = np.ones_like(deviation)

        probabilities = weights / weights.sum()
        rng = np.random.default_rng(seed)
        missing_indices = rng.choice(
            available_indices, n_to_add, replace=False, p=probabilities
        )

    # Set selected indices to NaN
    data_copy.iloc[missing_indices] = np.nan

    print(
        f"✓ Applied MNAR: {int(data_copy.isna().sum())}/{len(data)} values "
        f"({missing_rate*100:.1f}% target) missing"
    )
    print("  (Missing probability depends on the subsequently hidden value)")

    return data_copy


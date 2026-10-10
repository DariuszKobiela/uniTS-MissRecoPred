"""
SARIMAX Imputation
Uses SARIMA model to impute missing values based on time series patterns.
Uses Kalman smoothing to estimate missing values.
"""

import pandas as pd
import numpy as np
import warnings


_MAX_EXTRAPOLATION_RANGES = 100.0


def _likelihood_is_finite(model, params: np.ndarray) -> bool:
    try:
        value = model.loglike(np.asarray(params, dtype=float))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return False
    return bool(np.isfinite(value))


def _starting_parameters(model, observed: np.ndarray) -> np.ndarray:
    """Return start values whose Kalman likelihood is finite.

    Hannan–Rissanen starts estimated under heavy missingness can be explosive.
    The filter likelihood is then NaN, the optimizer exits before the first
    iteration, and the smoother state is entirely NaN.
    """
    start = np.asarray(model.start_params, dtype=float).copy()
    if _likelihood_is_finite(model, start):
        return start

    finite = observed[np.isfinite(observed)]
    differences = np.diff(finite) if finite.size > 1 else np.array([])
    innovation_variance = float(np.var(differences)) if differences.size else 1.0
    if not np.isfinite(innovation_variance) or innovation_variance <= 0:
        innovation_variance = 1.0
    safe = np.zeros_like(start)
    safe[-1] = innovation_variance
    warnings.warn(
        "SARIMAX default start parameters have a non-finite likelihood; "
        "refitting from a zero ARMA start.",
        RuntimeWarning,
        stacklevel=2,
    )
    return safe


def _linear_fallback(data: pd.Series, reason: str) -> pd.Series:
    """Return a bounded, deterministic fallback while preserving observations."""
    warnings.warn(
        f"SARIMAX reconstruction was unstable ({reason}); using linear interpolation fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    result = data.interpolate(method="linear", limit_direction="both")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("SARIMAX fallback could not produce finite values")
    return result


def _imputed_values_are_plausible(
    observed: np.ndarray, imputed: np.ndarray
) -> bool:
    """Reject finite numerical explosions that a simple finite check misses."""
    finite_observed = observed[np.isfinite(observed)]
    imputed = np.asarray(imputed, dtype=float)
    if finite_observed.size == 0 or not np.isfinite(imputed).all():
        return False

    observed_min = float(np.min(finite_observed))
    observed_max = float(np.max(finite_observed))
    observed_range = observed_max - observed_min
    scale = max(observed_range, float(np.std(finite_observed)), 1.0)
    margin = _MAX_EXTRAPOLATION_RANGES * scale
    return bool(
        np.all(imputed >= observed_min - margin)
        and np.all(imputed <= observed_max + margin)
    )


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
    observed = data_with_nan.to_numpy(dtype=float)
    
    # Suppress frequency warning (not needed for imputation, only for forecasting)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=Warning, message='.*frequency.*')
        
        # Fit SARIMAX model - it will use Kalman filter to handle missing values
        model = SARIMAX(data_with_nan, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        start = _starting_parameters(model, observed)
        try:
            fitted = model.fit(disp=False, start_params=start)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return _linear_fallback(data, f"fit failed: {exc}")

    if not bool(getattr(fitted, "mle_retvals", {}).get("converged", True)):
        return _linear_fallback(data, "optimizer did not converge")
    
    # Use smoothed states to get estimates for missing values
    # The smoothed states provide optimal estimates given all data
    smoothed = fitted.smoother_results.smoothed_state[0]
    
    # Create result series
    result = data.copy()
    
    # Fill missing values with smoothed estimates
    result[missing_mask] = smoothed[missing_mask]

    imputed = result[missing_mask].to_numpy(dtype=float)
    if not _imputed_values_are_plausible(observed, imputed):
        return _linear_fallback(data, "smoothed values were non-finite or implausible")
    
    return result


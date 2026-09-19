"""Cross missingness mechanisms with temporal missing-value structures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.experiment_naming import MISSINGNESS_STRUCTURES


def mechanism_weights(series: pd.Series, mechanism: str) -> np.ndarray:
    """Return positive per-position sampling weights for MCAR, MAR, or MNAR."""
    mechanism = mechanism.upper()
    n = len(series)
    if mechanism == "MCAR":
        return np.ones(n, dtype=np.float64)
    if mechanism == "MAR":
        # Time is a fully observed covariate, including after a value is hidden.
        return np.linspace(0.1, 1.0, n, dtype=np.float64)
    if mechanism == "MNAR":
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        observed = np.isfinite(values)
        if not observed.any():
            return np.ones(n, dtype=np.float64)
        median = float(np.median(values[observed]))
        deviation = np.zeros(n, dtype=np.float64)
        deviation[observed] = np.abs(values[observed] - median)
        scale = float(deviation[observed].max())
        if scale <= 0.0:
            return np.ones(n, dtype=np.float64)
        return 0.05 + deviation / scale
    raise ValueError(f"Unknown missingness mechanism: {mechanism}")


def _weighted_choice(
    candidates: np.ndarray, weights: np.ndarray, rng: np.random.Generator
) -> int:
    candidate_weights = np.asarray(weights[candidates], dtype=np.float64)
    total = float(candidate_weights.sum())
    probabilities = None if total <= 0.0 else candidate_weights / total
    return int(rng.choice(candidates, p=probabilities))


def _sample_points(
    available: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    isolate: bool,
) -> np.ndarray:
    selected = np.zeros(len(available), dtype=bool)
    for _ in range(count):
        candidates = np.flatnonzero(available & ~selected)
        if isolate:
            left_free = np.r_[True, ~(selected | ~available)[:-1]]
            right_free = np.r_[~(selected | ~available)[1:], True]
            isolated = candidates[left_free[candidates] & right_free[candidates]]
            if len(isolated):
                candidates = isolated
        if not len(candidates):
            raise ValueError("Cannot place the requested scattered missing values")
        selected[_weighted_choice(candidates, weights, rng)] = True
    return selected


def _partition_block_lengths(
    total: int, minimum: int, maximum: int, rng: np.random.Generator
) -> list[int]:
    if total <= 0:
        return []
    minimum = max(2, int(minimum))
    maximum = max(minimum, int(maximum))
    if total < minimum:
        return [total]
    lengths: list[int] = []
    remaining = total
    while remaining:
        if remaining <= maximum:
            lengths.append(remaining)
            break
        upper = min(maximum, remaining - minimum)
        length = int(rng.integers(minimum, upper + 1))
        lengths.append(length)
        remaining -= length
    return sorted(lengths, reverse=True)


def _sample_blocks(
    available: np.ndarray,
    weights: np.ndarray,
    total: int,
    rng: np.random.Generator,
    *,
    min_block_length: int,
    max_block_length: int,
) -> np.ndarray:
    selected = np.zeros(len(available), dtype=bool)
    lengths = _partition_block_lengths(
        total, min_block_length, max_block_length, rng
    )
    for length in lengths:
        starts: list[int] = []
        start_weights: list[float] = []
        for start in range(0, len(available) - length + 1):
            stop = start + length
            if not available[start:stop].all() or selected[start:stop].any():
                continue
            # A one-sample observed separator makes realized runs unambiguous.
            if start > 0 and (selected[start - 1] or not available[start - 1]):
                continue
            if stop < len(available) and (selected[stop] or not available[stop]):
                continue
            starts.append(start)
            start_weights.append(float(weights[start:stop].mean()))
        if not starts:
            raise ValueError(
                "Cannot place non-overlapping contiguous gaps with the requested "
                "rate/block constraints"
            )
        probabilities = np.asarray(start_weights, dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        chosen = int(rng.choice(np.asarray(starts), p=probabilities))
        selected[chosen : chosen + length] = True
    return selected


def apply_structured_missingness(
    data: pd.Series,
    missing_rate: float,
    mechanism: str,
    structure: str,
    seed: int | None = None,
    *,
    min_block_length: int = 3,
    max_block_length: int = 24,
    mixed_scattered_fraction: float = 0.5,
) -> pd.Series:
    """Apply an exact-rate mechanism × structure missingness realization."""
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be between 0 and 1, got {missing_rate}")
    structure = structure.lower()
    if structure not in MISSINGNESS_STRUCTURES:
        raise ValueError(f"Unknown missingness structure: {structure}")
    if not 0.0 <= mixed_scattered_fraction <= 1.0:
        raise ValueError("mixed_scattered_fraction must be between 0 and 1")

    result = data.copy()
    target_total = int(len(result) * missing_rate)
    existing = result.isna().to_numpy()
    to_add = max(0, target_total - int(existing.sum()))
    if to_add == 0:
        return result

    available = ~existing
    if to_add > int(available.sum()):
        raise ValueError("Requested missing rate exceeds the number of observed values")
    weights = mechanism_weights(result, mechanism)
    rng = np.random.default_rng(seed)

    if structure == "scattered":
        selected = _sample_points(available, weights, to_add, rng, isolate=False)
    elif structure == "contiguous":
        selected = _sample_blocks(
            available,
            weights,
            to_add,
            rng,
            min_block_length=min_block_length,
            max_block_length=max_block_length,
        )
    else:
        if to_add < 3:
            selected = _sample_points(available, weights, to_add, rng, isolate=False)
        else:
            point_count = int(round(to_add * mixed_scattered_fraction))
            point_count = min(max(1, point_count), to_add - 2)
            block_count = to_add - point_count
            selected = _sample_blocks(
                available,
                weights,
                block_count,
                rng,
                min_block_length=min_block_length,
                max_block_length=max_block_length,
            )
            remaining_available = available & ~selected
            points = _sample_points(
                remaining_available, weights, point_count, rng, isolate=True
            )
            selected |= points

    if int(selected.sum()) != to_add:
        raise RuntimeError(
            f"Internal missingness error: selected {selected.sum()} instead of {to_add}"
        )
    result.iloc[np.flatnonzero(selected)] = np.nan
    return result

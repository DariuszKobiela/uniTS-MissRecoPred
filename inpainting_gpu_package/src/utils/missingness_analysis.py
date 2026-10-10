"""Gap-level and realization-level missingness diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def infer_sampling_interval_seconds(index: pd.Index) -> tuple[float | None, bool]:
    """Infer median positive datetime interval and whether sampling is irregular."""
    if isinstance(index, pd.DatetimeIndex):
        parsed = index
    elif pd.api.types.is_numeric_dtype(index.dtype):
        return None, False
    else:
        parsed = pd.to_datetime(index, errors="coerce")
        if parsed.isna().any():
            return None, False
    if len(parsed) < 2:
        return None, False
    deltas = np.diff(parsed.asi8).astype(np.float64) / 1_000_000_000.0
    deltas = deltas[deltas > 0]
    if not len(deltas):
        return None, False
    median = float(np.median(deltas))
    irregular = not bool(np.allclose(deltas, median, rtol=1e-6, atol=1e-9))
    return median, irregular


def gap_records(mask: np.ndarray, index: pd.Index) -> list[dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(index):
        raise ValueError("Mask and index lengths differ")
    interval_seconds, irregular = infer_sampling_interval_seconds(index)
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    records: list[dict[str, Any]] = []
    for gap_number, (start, stop) in enumerate(zip(starts, stops), start=1):
        length = int(stop - start)
        span_seconds = None
        if interval_seconds is not None:
            parsed = pd.to_datetime(index, errors="coerce")
            span_seconds = float((parsed[stop - 1] - parsed[start]).total_seconds())
        records.append(
            {
                "gap_number": gap_number,
                "start_position": int(start),
                "end_position": int(stop - 1),
                "start_index": str(index[start]),
                "end_index": str(index[stop - 1]),
                "length_samples": length,
                "span_seconds": span_seconds,
                "coverage_seconds": (
                    None
                    if interval_seconds is None
                    else float(span_seconds + interval_seconds)
                ),
                "is_singleton": length == 1,
                "is_edge_gap": start == 0 or stop == len(mask),
                "sampling_interval_seconds": interval_seconds,
                "irregular_sampling": irregular,
            }
        )
    return records


def _stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
            f"{prefix}_max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def summarize_missingness(
    series: pd.Series, *, metadata: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a realization summary and one record per contiguous NaN run."""
    mask = series.isna().to_numpy()
    gaps = gap_records(mask, series.index)
    n_gaps = len(gaps)
    summary: dict[str, Any] = dict(metadata or {})
    summary.update(
        {
            "n_total": int(len(series)),
            "n_missing": int(mask.sum()),
            "actual_missing_rate": float(mask.mean()) if len(mask) else 0.0,
            "n_gaps": n_gaps,
            "singleton_gap_percent": (
                100.0 * sum(bool(g["is_singleton"]) for g in gaps) / n_gaps
                if n_gaps
                else 0.0
            ),
            "sampling_interval_seconds": (
                gaps[0]["sampling_interval_seconds"] if gaps else None
            ),
            "irregular_sampling": gaps[0]["irregular_sampling"] if gaps else False,
        }
    )
    summary.update(_stats([g["length_samples"] for g in gaps], "gap_length_samples"))
    summary.update(
        _stats(
            [g["coverage_seconds"] for g in gaps if g["coverage_seconds"] is not None],
            "gap_coverage_seconds",
        )
    )
    enriched = [{**(metadata or {}), **gap} for gap in gaps]
    return summary, enriched

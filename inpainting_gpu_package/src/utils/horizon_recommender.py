"""
Forecast horizon recommendation utilities.

Infers the actual sampling interval of a cleaned series (mean of the most
common gap when irregular) and computes H_short / H_long dynamically from
that interval plus the two holdout constraints:

    H_long / n  <=  0.20
    n - H_long  ≳  200

There is no static frequency-group table. If a candidate is Unsafe, H_long
is shortened until the status is at least Safe (or H_long = 1 if the series
is too short).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STATUS_IDEAL = "Ideal"
STATUS_SAFE = "Safe"
STATUS_MAXIMUM_SAFE = "Maximum safe"
STATUS_UNSAFE = "Unsafe"

SAFE_STATUSES = {STATUS_IDEAL, STATUS_SAFE, STATUS_MAXIMUM_SAFE}

# Calendar cycle lengths in seconds. These are physical units used to turn an
# inferred sampling interval into a number of samples — not a lookup of H.
# Calendar lengths used only to label horizon spans in reports, not to pick H.


@dataclass(frozen=True)
class HorizonConstraints:
    max_holdout_share: float = 0.20
    min_train_length: int = 200
    ideal_holdout_share: float = 0.15
    unsafe_train_threshold: int = 190
    h_short: int = 12
    h_long: int = 96
    short_series_h_short: int = 12
    short_series_h_long: int = 24


@dataclass
class SeriesProfile:
    series_id: str
    n: int
    index_type: str
    sampling_interval_seconds: Optional[float]
    sampling_label: str
    inferred_freq: Optional[str]
    mean_delta_seconds: Optional[float]
    median_delta_seconds: Optional[float]
    freq_confidence: float
    is_irregular: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class SeriesHorizonRecommendation:
    series_id: str
    sampling_interval_seconds: Optional[float]
    sampling_label: str
    inferred_freq: Optional[str]
    mean_delta_seconds: Optional[float]
    median_delta_seconds: Optional[float]
    freq_confidence: float
    is_irregular: bool
    n: int
    trimmed_length: int
    h_short: int
    h_long: int
    horizons: List[int]
    train_length: int
    holdout_share: float
    status: str
    short_cycle_label: str
    long_cycle_label: str
    notes: List[str] = field(default_factory=list)


def _delta_seconds(deltas: pd.Series) -> np.ndarray:
    secs = deltas.dt.total_seconds().to_numpy(dtype=float)
    return secs[np.isfinite(secs) & (secs > 0)]


_DURATION_UNITS: Tuple[Tuple[str, str, float], ...] = (
    ("year", "years", 365.25 * 86400.0),
    ("month", "months", 30.4375 * 86400.0),
    ("week", "weeks", 7.0 * 86400.0),
    ("d", "d", 86400.0),
    ("h", "h", 3600.0),
    ("min", "min", 60.0),
    ("s", "s", 1.0),
)


def format_interval(seconds: float) -> str:
    """Human-readable duration for an inferred sampling interval."""
    return format_duration_aggregated(seconds)


def format_duration_aggregated(seconds: float) -> str:
    """
    Express a duration in the most aggregated unit that still reads cleanly:

    s → min → h → d → weeks → months → years
    """
    if seconds is None or seconds <= 0 or not np.isfinite(seconds):
        return "—"

    for singular, plural, size in _DURATION_UNITS:
        value = seconds / size
        nearly_int = abs(value - round(value)) < 0.08 * max(value, 1.0)
        if value >= 1.5 or (value >= 1.0 and nearly_int):
            if nearly_int:
                rounded = int(round(value))
                unit = singular if rounded == 1 else plural
                return f"{rounded} {unit}"
            return f"{value:.3g} {plural}"
    return f"{seconds:.3g} s"


def format_horizon_span(n_samples: int, interval_seconds: Optional[float]) -> str:
    """Business-readable length of a horizon in samples."""
    if not interval_seconds or interval_seconds <= 0 or n_samples <= 0:
        return "—"
    return format_duration_aggregated(n_samples * interval_seconds)


def _adopted_interval(secs: np.ndarray) -> Tuple[float, float, float, float, str]:
    """
    Return (adopted, mean, median, confidence, method).

    For irregular series the adopted interval is the mean of the most common
    gap cluster, not a snap to a named frequency (15 min, hourly, …).
    """
    mean_sec = float(np.mean(secs))
    median_sec = float(np.median(secs))
    unit = 1.0 if median_sec >= 2.0 else 0.1
    rounded = np.round(secs / unit) * unit
    values, counts = np.unique(rounded, return_counts=True)
    mode = float(values[int(np.argmax(counts))])
    cluster = secs[np.abs(rounded - mode) <= (unit * 0.51)]
    if len(cluster) == 0:
        cluster = secs
        method = "mean of all intervals"
        adopted = mean_sec
    else:
        adopted = float(np.mean(cluster))
        share = len(cluster) / len(secs)
        if share >= 0.5:
            method = "mean of most common interval"
        else:
            adopted = mean_sec
            method = "mean of all intervals"

    tol = max(abs(adopted) * 0.20, unit)
    confidence = float(np.mean(np.abs(secs - adopted) <= tol))
    return adopted, mean_sec, median_sec, confidence, method


def infer_series_profile(series_id: str, index: pd.Index) -> SeriesProfile:
    """Infer the actual sampling interval; do not snap to a named frequency."""
    notes: List[str] = []
    n = len(index)

    if not isinstance(index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(index):
            return SeriesProfile(
                series_id=series_id,
                n=n,
                index_type="numeric",
                sampling_interval_seconds=None,
                sampling_label="unknown (numeric index)",
                inferred_freq=None,
                mean_delta_seconds=None,
                median_delta_seconds=None,
                freq_confidence=0.0,
                is_irregular=True,
                notes=["Non-datetime index; horizons from holdout/train constraints only."],
            )
        try:
            dt_index = pd.to_datetime(index, errors="coerce")
            if isinstance(dt_index, pd.DatetimeIndex) and dt_index.notna().all():
                index = dt_index
            else:
                return SeriesProfile(
                    series_id=series_id,
                    n=n,
                    index_type="numeric",
                    sampling_interval_seconds=None,
                    sampling_label="unknown (non-datetime index)",
                    inferred_freq=None,
                    mean_delta_seconds=None,
                    median_delta_seconds=None,
                    freq_confidence=0.0,
                    is_irregular=True,
                    notes=["Non-datetime index; horizons from holdout/train constraints only."],
                )
        except (ValueError, TypeError):
            return SeriesProfile(
                series_id=series_id,
                n=n,
                index_type="numeric",
                sampling_interval_seconds=None,
                sampling_label="unknown (non-datetime index)",
                inferred_freq=None,
                mean_delta_seconds=None,
                median_delta_seconds=None,
                freq_confidence=0.0,
                is_irregular=True,
                notes=["Non-datetime index; horizons from holdout/train constraints only."],
            )

    inferred_freq: Optional[str] = None
    if len(index) >= 3:
        try:
            inferred_freq = pd.infer_freq(index)
        except (ValueError, TypeError):
            inferred_freq = None

    if len(index) < 2:
        return SeriesProfile(
            series_id=series_id,
            n=n,
            index_type="datetime",
            sampling_interval_seconds=None,
            sampling_label="unknown (too short)",
            inferred_freq=inferred_freq,
            mean_delta_seconds=None,
            median_delta_seconds=None,
            freq_confidence=0.0,
            is_irregular=True,
            notes=["Series too short to infer a sampling interval."],
        )

    deltas = index.to_series().diff().iloc[1:]
    secs = _delta_seconds(deltas)
    if len(secs) == 0:
        return SeriesProfile(
            series_id=series_id,
            n=n,
            index_type="datetime",
            sampling_interval_seconds=None,
            sampling_label="unknown (no positive deltas)",
            inferred_freq=inferred_freq,
            mean_delta_seconds=None,
            median_delta_seconds=None,
            freq_confidence=0.0,
            is_irregular=True,
            notes=["No positive time deltas found."],
        )

    adopted, mean_sec, median_sec, confidence, method = _adopted_interval(secs)
    is_irregular = confidence < 0.60
    label = f"every {format_interval(adopted)}"
    if is_irregular:
        notes.append(
            f"Irregular sampling (confidence={confidence:.2f}); "
            f"adopted {method} = {format_interval(adopted)}."
        )
    else:
        notes.append(f"Regular sampling; interval {format_interval(adopted)}.")

    return SeriesProfile(
        series_id=series_id,
        n=n,
        index_type="datetime",
        sampling_interval_seconds=adopted,
        sampling_label=label,
        inferred_freq=inferred_freq,
        mean_delta_seconds=mean_sec,
        median_delta_seconds=median_sec,
        freq_confidence=confidence,
        is_irregular=is_irregular,
        notes=notes,
    )


def _holdout_ok(n: int, h_long: int, constraints: HorizonConstraints) -> bool:
    return n > 0 and 0 < h_long < n and (h_long / n) <= constraints.max_holdout_share + 1e-12


def _primary_pair_fits(n: int, constraints: HorizonConstraints) -> bool:
    h_long = constraints.h_long
    if not _holdout_ok(n, h_long, constraints):
        return False
    return (n - h_long) >= constraints.min_train_length


def choose_sample_horizons(
    n: int,
    constraints: HorizonConstraints,
) -> Tuple[int, int, str]:
    """
    Literature pair is H_short=12, H_long=96.

    If that pair violates holdout ≤ 20% or training length ≳ 200, use the
    short-series pair H_short=12, H_long=24 (same 1:2 proportion, smaller long
    window). Only if 24 still exceeds 20% of n is H_long capped further.
    """
    if _primary_pair_fits(n, constraints):
        return (
            constraints.h_short,
            constraints.h_long,
            f"literature pair H_short={constraints.h_short}, H_long={constraints.h_long}",
        )

    h_short = constraints.short_series_h_short
    h_long = constraints.short_series_h_long
    if _holdout_ok(n, h_long, constraints):
        return (
            h_short,
            h_long,
            f"short-series pair H_short={h_short}, H_long={h_long} "
            f"(n={n} cannot host H_long={constraints.h_long})",
        )

    max_by_share = max(1, min(n - 1, int(np.floor(constraints.max_holdout_share * n))))
    h_long = min(h_long, max_by_share)
    h_short = max(1, min(h_short, h_long // 2 if h_long > 1 else 1))
    return (
        h_short,
        h_long,
        f"capped to H_long={h_long} so H_long/n ≤ {constraints.max_holdout_share:.0%}",
    )


def classify_status(n: int, h_long: int, constraints: HorizonConstraints) -> str:
    if n <= 0 or h_long <= 0:
        return STATUS_UNSAFE

    holdout_share = h_long / n
    train_left = n - h_long

    if holdout_share > constraints.max_holdout_share + 1e-12:
        return STATUS_UNSAFE
    if train_left < constraints.unsafe_train_threshold:
        return STATUS_UNSAFE

    near_floor = (
        constraints.min_train_length - 5
        <= train_left
        <= constraints.min_train_length + 5
    )
    if near_floor and holdout_share <= constraints.max_holdout_share:
        return STATUS_MAXIMUM_SAFE

    if (
        train_left >= constraints.min_train_length
        and holdout_share < constraints.ideal_holdout_share
    ):
        return STATUS_IDEAL

    return STATUS_SAFE


def _max_h_long_for_safe(n: int, constraints: HorizonConstraints) -> int:
    """Largest H_long that can still be Safe/Ideal, or 1 if impossible."""
    max_by_share = int(np.floor(constraints.max_holdout_share * n))
    max_h = min(max_by_share, n - 1)
    if max_h < 1:
        return 1

    if n > constraints.min_train_length:
        max_h = min(max_h, n - constraints.min_train_length)
    elif n > constraints.unsafe_train_threshold:
        max_h = min(max_h, n - constraints.unsafe_train_threshold)

    return max(1, max_h)


def fit_horizons_to_constraints(
    n: int,
    h_short: int,
    h_long: int,
    constraints: HorizonConstraints,
) -> Tuple[int, int, str, List[str]]:
    """Apply holdout/train inequalities; shorten Unsafe candidates to Safe."""
    notes: List[str] = []
    original_long = h_long

    max_safe = _max_h_long_for_safe(n, constraints)
    if h_long > max_safe:
        notes.append(
            f"Reduced H_long from {original_long} to {max_safe} "
            f"(n={n}, H_long/n ≤ {constraints.max_holdout_share:.0%}, "
            f"n − H_long ≳ {constraints.min_train_length})."
        )
        h_long = max_safe

    if h_long >= n:
        h_long = max(1, n - 1)
        notes.append(f"Clamped H_long to {h_long} because n={n}.")

    h_short = max(1, min(h_short, h_long // 2 if h_long > 1 else 1))

    status = classify_status(n, h_long, constraints)
    while status == STATUS_UNSAFE and h_long > 1:
        h_long -= 1
        h_short = max(1, min(h_short, h_long // 2 if h_long > 1 else 1))
        status = classify_status(n, h_long, constraints)
        if status != STATUS_UNSAFE:
            notes.append(f"Shortened H_long to {h_long} so status became {status}.")

    if status == STATUS_UNSAFE:
        notes.append(
            f"Series too short for a Safe holdout (n={n}; "
            f"need n − H_long ≳ {constraints.unsafe_train_threshold})."
        )

    return h_short, h_long, status, notes


def resolve_experiment_horizons(
    series_id: str,
    experiment_horizons: Optional[Mapping[str, Sequence[int]]],
) -> Optional[List[int]]:
    """Match a series filename to an experiment horizon list."""
    if not experiment_horizons:
        return None
    if series_id in experiment_horizons:
        return [int(h) for h in experiment_horizons[series_id]]
    stem = Path(series_id).stem.lower()
    name = series_id.lower()
    for key, values in experiment_horizons.items():
        token = str(key).lower()
        stem_key = Path(str(key)).stem.lower()
        if token in name or stem_key in stem or stem in stem_key:
            return [int(h) for h in values]
    return None


def recommend_horizons_for_profile(
    profile: SeriesProfile,
    constraints: Optional[HorizonConstraints] = None,
    experiment_horizons: Optional[Mapping[str, Sequence[int]]] = None,
) -> SeriesHorizonRecommendation:
    cons = constraints or HorizonConstraints()
    notes = list(profile.notes)

    forced = resolve_experiment_horizons(profile.series_id, experiment_horizons)
    if forced:
        horizons = sorted({int(h) for h in forced if int(h) > 0})
        h_short = min(horizons)
        h_long = max(horizons)
        notes.append(
            f"Experiment horizons {horizons}; split uses longest H={h_long}."
        )
        status = classify_status(profile.n, h_long, cons)
        if h_long >= profile.n:
            status = STATUS_UNSAFE
            notes.append(f"Longest horizon {h_long} is not shorter than n={profile.n}.")
        elif _holdout_ok(profile.n, h_long, cons) and status == STATUS_UNSAFE:
            status = STATUS_MAXIMUM_SAFE
        short_label = format_horizon_span(h_short, profile.sampling_interval_seconds)
        long_label = format_horizon_span(h_long, profile.sampling_interval_seconds)
    else:
        h_short, h_long, rule = choose_sample_horizons(profile.n, cons)
        notes.append(rule)
        horizons = sorted({h_short, h_long})
        used_short_pair = (
            h_long == cons.short_series_h_long and h_short == cons.short_series_h_short
        )
        short_label = format_horizon_span(h_short, profile.sampling_interval_seconds)
        long_label = format_horizon_span(h_long, profile.sampling_interval_seconds)

        if used_short_pair and _holdout_ok(profile.n, h_long, cons):
            status = STATUS_MAXIMUM_SAFE
        elif _primary_pair_fits(profile.n, cons) and h_long == cons.h_long:
            status = classify_status(profile.n, h_long, cons)
        else:
            h_short, h_long, status, fit_notes = fit_horizons_to_constraints(
                profile.n, h_short, h_long, cons
            )
            horizons = sorted({h_short, h_long})
            short_label = format_horizon_span(h_short, profile.sampling_interval_seconds)
            long_label = format_horizon_span(h_long, profile.sampling_interval_seconds)
            notes.extend(fit_notes)

    train_length = profile.n - h_long
    holdout_share = h_long / profile.n if profile.n else 0.0

    return SeriesHorizonRecommendation(
        series_id=profile.series_id,
        sampling_interval_seconds=profile.sampling_interval_seconds,
        sampling_label=profile.sampling_label,
        inferred_freq=profile.inferred_freq,
        mean_delta_seconds=profile.mean_delta_seconds,
        median_delta_seconds=profile.median_delta_seconds,
        freq_confidence=profile.freq_confidence,
        is_irregular=profile.is_irregular,
        n=profile.n,
        trimmed_length=profile.n,
        h_short=h_short,
        h_long=h_long,
        horizons=horizons,
        train_length=train_length,
        holdout_share=holdout_share,
        status=status,
        short_cycle_label=short_label,
        long_cycle_label=long_label,
        notes=notes,
    )


def analyze_dataframe(
    series_id: str,
    df: pd.DataFrame,
    *,
    constraints: Optional[HorizonConstraints] = None,
    experiment_horizons: Optional[Mapping[str, Sequence[int]]] = None,
) -> Tuple[SeriesProfile, SeriesHorizonRecommendation]:
    profile = infer_series_profile(series_id, df.index)
    rec = recommend_horizons_for_profile(
        profile,
        constraints=constraints,
        experiment_horizons=experiment_horizons,
    )
    return profile, rec


def analyze_profiles_batch(
    profiles: Sequence[SeriesProfile],
    *,
    constraints: Optional[HorizonConstraints] = None,
    experiment_horizons: Optional[Mapping[str, Sequence[int]]] = None,
) -> List[SeriesHorizonRecommendation]:
    cons = constraints or HorizonConstraints()
    return [
        recommend_horizons_for_profile(
            p,
            constraints=cons,
            experiment_horizons=experiment_horizons,
        )
        for p in profiles
    ]


def metadata_document(
    series_recs: Sequence[SeriesHorizonRecommendation],
    *,
    input_dir: str,
    output_dir: str,
    constraints: HorizonConstraints,
    generated_at: str,
    split_entries: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.2",
        "generated_at": generated_at,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "constraints": {
            "max_holdout_share": constraints.max_holdout_share,
            "min_train_length": constraints.min_train_length,
            "ideal_holdout_share": constraints.ideal_holdout_share,
            "unsafe_train_threshold": constraints.unsafe_train_threshold,
            "h_short": constraints.h_short,
            "h_long": constraints.h_long,
            "short_series_h_short": constraints.short_series_h_short,
            "short_series_h_long": constraints.short_series_h_long,
        },
        "series": {
            rec.series_id: {
                "sampling_interval_seconds": rec.sampling_interval_seconds,
                "sampling_label": rec.sampling_label,
                "inferred_freq": rec.inferred_freq,
                "mean_delta_seconds": rec.mean_delta_seconds,
                "median_delta_seconds": rec.median_delta_seconds,
                "freq_confidence": round(rec.freq_confidence, 4),
                "is_irregular": rec.is_irregular,
                "n": rec.n,
                "trimmed_length": rec.trimmed_length,
                "h_short": rec.h_short,
                "h_long": rec.h_long,
                "horizons": rec.horizons,
                "train_length": rec.train_length,
                "holdout_share": round(rec.holdout_share, 6),
                "status": rec.status,
                "short_cycle_label": rec.short_cycle_label,
                "long_cycle_label": rec.long_cycle_label,
                "notes": rec.notes,
                **(
                    dict(split_entries.get(rec.series_id, {}))
                    if split_entries and rec.series_id in split_entries
                    else {}
                ),
            }
            for rec in series_recs
        },
    }


def load_h_long_lookup(metadata: Mapping[str, Any]) -> Dict[str, int]:
    """Extract per-series longest holdout (max horizon) from metadata."""
    series = metadata.get("series", {})
    lookup: Dict[str, int] = {}
    for series_id, payload in series.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("horizons"):
            lookup[series_id] = int(max(int(h) for h in payload["horizons"]))
        elif "h_long" in payload:
            lookup[series_id] = int(payload["h_long"])
    return lookup

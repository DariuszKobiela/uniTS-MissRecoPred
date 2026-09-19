"""Three-way temporal split planning for rebuttal experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from utils.horizon_recommender import HorizonConstraints


@dataclass(frozen=True)
class SplitBoundaries:
    """Index boundaries for reconstruction-train, SD2 validation, and rolling test."""

    series_length: int
    h_max: int
    n_origins: int
    reconstruction_end: int
    validation_end: int
    test_length: int
    validation_length: int
    reconstruction_length: int
    test_start: int
    validation_start: int
    max_holdout_share: float
    notes: tuple[str, ...] = ()

    @property
    def test_end(self) -> int:
        return self.series_length


def resolve_n_origins(dataset_name: str, defaults: Mapping[str, int], fallback: int = 5) -> int:
    """Resolve per-dataset rolling-origin count."""
    stem = dataset_name.replace(".csv", "")
    for key, value in defaults.items():
        token = str(key).replace(".csv", "")
        if token == stem or token in dataset_name:
            return int(value)
    return int(fallback)


def min_rolling_test_length(h_max: int, n_origins: int) -> int:
    """Minimum test holdout length for ``H_max`` and ``n_origins`` evenly spread origins."""
    if h_max < 1:
        raise ValueError("h_max must be positive")
    if n_origins < 1:
        raise ValueError("n_origins must be positive")
    return max(h_max, h_max + n_origins - 1)


def largest_safe_rolling_test_length(
    series_length: int,
    h_max: int,
    n_origins: int,
    constraints: HorizonConstraints | None = None,
    min_reconstruction_length: int = 200,
    min_validation_length: int = 64,
) -> tuple[int, tuple[str, ...]]:
    """Largest holdout up to 20% that still fits ``H_max`` and origin spacing."""
    cons = constraints or HorizonConstraints()
    notes: list[str] = []
    n = int(series_length)
    if n < 1:
        raise ValueError("series_length must be positive")

    min_test = min_rolling_test_length(h_max, n_origins)
    max_by_share = int(np.floor(cons.max_holdout_share * n))
    max_by_train = n - min_reconstruction_length - min_validation_length
    max_test = min(max_by_share, max_by_train)

    if max_test < min_test:
        raise ValueError(
            f"Series length n={n} cannot host test holdout min={min_test} "
            f"(H_max={h_max}, n_origins={n_origins}) with "
            f"max_share={cons.max_holdout_share:.0%}, "
            f"min_reconstruction={min_reconstruction_length}, "
            f"min_validation={min_validation_length}."
        )

    test_length = max_test
    if test_length > min_test:
        notes.append(
            f"Using largest safe holdout {test_length} ({test_length / n:.1%} of n) "
            f"above minimum {min_test} for H_max={h_max} and {n_origins} origins."
        )
    else:
        notes.append(
            f"Using minimum holdout {test_length} for H_max={h_max} and {n_origins} origins."
        )
    return test_length, tuple(notes)


def compute_validation_length(
    remaining_length: int,
    *,
    share: float = 0.10,
    min_samples: int = 64,
    max_samples: int = 500,
) -> int:
    """Validation block length taken from the tail of the reconstruction region."""
    if remaining_length < min_samples:
        raise ValueError(
            f"Not enough samples ({remaining_length}) for SD2 validation minimum {min_samples}."
        )
    target = int(np.floor(share * remaining_length))
    target = max(min_samples, target)
    target = min(max_samples, target, remaining_length - 1)
    return max(min_samples, target)


def plan_three_way_split(
    series_length: int,
    h_max: int,
    n_origins: int,
    *,
    constraints: HorizonConstraints | None = None,
    validation_share: float = 0.10,
    validation_min_samples: int = 64,
    validation_max_samples: int = 500,
    min_reconstruction_length: int = 200,
) -> SplitBoundaries:
    """Plan reconstruction-train, SD2 validation, and rolling-test index ranges."""
    cons = constraints or HorizonConstraints()
    notes: list[str] = []

    test_length, test_notes = largest_safe_rolling_test_length(
        series_length,
        h_max,
        n_origins,
        constraints=cons,
        min_reconstruction_length=min_reconstruction_length,
        min_validation_length=validation_min_samples,
    )
    notes.extend(test_notes)

    pre_test = series_length - test_length
    validation_length = compute_validation_length(
        pre_test,
        share=validation_share,
        min_samples=validation_min_samples,
        max_samples=validation_max_samples,
    )
    reconstruction_length = pre_test - validation_length
    if reconstruction_length < min_reconstruction_length:
        raise ValueError(
            f"Reconstruction train length {reconstruction_length} < {min_reconstruction_length} "
            f"after reserving test={test_length} and validation={validation_length}."
        )

    notes.append(
        f"Three-way split: reconstruction[0:{reconstruction_length}], "
        f"sd2_validation[{reconstruction_length}:{reconstruction_length + validation_length}], "
        f"rolling_test[{series_length - test_length}:{series_length}]."
    )

    return SplitBoundaries(
        series_length=series_length,
        h_max=h_max,
        n_origins=n_origins,
        reconstruction_end=reconstruction_length,
        validation_end=reconstruction_length + validation_length,
        test_length=test_length,
        validation_length=validation_length,
        reconstruction_length=reconstruction_length,
        test_start=series_length - test_length,
        validation_start=reconstruction_length,
        max_holdout_share=cons.max_holdout_share,
        notes=tuple(notes),
    )


def split_manifest_entry(
    dataset_name: str,
    boundaries: SplitBoundaries,
    horizons: Sequence[int],
) -> dict:
    """JSON-serializable split record with disjoint index ranges."""
    return {
        "dataset": dataset_name,
        "series_length": boundaries.series_length,
        "h_max": boundaries.h_max,
        "horizons": [int(h) for h in horizons],
        "n_origins": boundaries.n_origins,
        "reconstruction_train": {
            "start": 0,
            "end_exclusive": boundaries.reconstruction_end,
            "length": boundaries.reconstruction_length,
        },
        "sd2_validation": {
            "start": boundaries.validation_start,
            "end_exclusive": boundaries.validation_end,
            "length": boundaries.validation_length,
        },
        "rolling_test": {
            "start": boundaries.test_start,
            "end_exclusive": boundaries.test_end,
            "length": boundaries.test_length,
        },
        "holdout_share": round(boundaries.test_length / boundaries.series_length, 6),
        "notes": list(boundaries.notes),
    }

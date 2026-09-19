"""Window planning and merging for local SD2 time-series reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconstructionWindow:
    """A context window and the non-overlapping core written to the result."""

    window_start: int
    window_stop: int
    core_start: int
    core_stop: int

    @property
    def window_length(self) -> int:
        return self.window_stop - self.window_start


def plan_reconstruction_windows(
    length: int,
    window_samples: int = 512,
    context_samples: int = 64,
) -> list[ReconstructionWindow]:
    """Plan overlapping context windows with disjoint output cores.

    A window has at most ``window_samples`` values. Interior cores contain
    ``window_samples - 2 * context_samples`` values. Only a core is committed
    to the final series, so every missing sample is reconstructed exactly once.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2")
    if context_samples < 0:
        raise ValueError("context_samples must be non-negative")
    if 2 * context_samples >= window_samples:
        raise ValueError("context_samples must be smaller than half the window")
    if length == 0:
        return []
    if length <= window_samples:
        return [ReconstructionWindow(0, length, 0, length)]

    core_size = window_samples - 2 * context_samples
    max_window_start = length - window_samples
    windows: list[ReconstructionWindow] = []
    for core_start in range(0, length, core_size):
        core_stop = min(core_start + core_size, length)
        desired_start = core_start - context_samples
        window_start = min(max(desired_start, 0), max_window_start)
        window_stop = window_start + window_samples
        windows.append(
            ReconstructionWindow(
                window_start=window_start,
                window_stop=window_stop,
                core_start=core_start,
                core_stop=core_stop,
            )
        )
    return windows


def reconstruct_in_windows(
    data: pd.Series,
    reconstruct_window: Callable[[pd.Series], pd.Series],
    window_samples: int = 512,
    context_samples: int = 64,
    *,
    progress_label: str | None = None,
) -> pd.Series:
    """Reconstruct missing values locally while preserving all observations."""
    missing = data.isna().to_numpy()
    if not missing.any():
        return data.copy()

    result = data.copy()
    plans = plan_reconstruction_windows(
        len(data),
        window_samples=window_samples,
        context_samples=context_samples,
    )
    active = [plan for plan in plans if missing[plan.core_start : plan.core_stop].any()]

    for number, plan in enumerate(active, start=1):
        if progress_label:
            print(
                f"  {progress_label}: window {number}/{len(active)} "
                f"[{plan.window_start}:{plan.window_stop}], "
                f"core [{plan.core_start}:{plan.core_stop}]"
            )

        local = data.iloc[plan.window_start : plan.window_stop].copy()
        reconstructed = reconstruct_window(local)
        if len(reconstructed) != len(local):
            raise ValueError(
                f"Window reconstructor changed the number of samples: {len(local)} -> {len(reconstructed)}"
            )

        absolute = np.arange(plan.core_start, plan.core_stop)
        targets = absolute[missing[plan.core_start : plan.core_stop]]
        local_positions = targets - plan.window_start
        result.iloc[targets] = reconstructed.iloc[local_positions].to_numpy()

    return result

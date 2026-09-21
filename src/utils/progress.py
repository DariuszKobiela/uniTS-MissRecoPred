"""Shared tqdm bars with remaining percentage and a time estimate."""

from __future__ import annotations

from typing import Any, Iterable, TypeVar

from tqdm import tqdm as _tqdm
from tqdm.auto import tqdm as _tqdm_auto

T = TypeVar("T")

# {l_bar} already includes "{desc}: {percentage:3.0f}%|".
# {elapsed}<{remaining} is the wall-clock ETA from the observed rate.
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def _defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.3)
    kwargs.setdefault("smoothing", 0.08)
    kwargs.setdefault("bar_format", BAR_FORMAT)
    kwargs.setdefault("disable", None)
    return kwargs


def tqdm(iterable: Iterable[T] | None = None, **kwargs: Any):
    """Progress bar that always shows percent remaining and ETA when total is known."""
    return _tqdm(iterable, **_defaults(kwargs))


def tqdm_auto(iterable: Iterable[T] | None = None, **kwargs: Any):
    """Notebook-aware variant used by the SD2 fine-tuning script."""
    return _tqdm_auto(iterable, **_defaults(kwargs))

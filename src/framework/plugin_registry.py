"""
Runtime registration and setuptools entry-point discovery for reconstruction and
prediction models. Third-party packages can expose models without editing this repo.

Entry point groups (declare under ``[project.entry-points]`` in *your* package):

- ``units_missrecopred.reconstruction`` — entry name = model id, value = ``module:callable``
- ``units_missrecopred.prediction`` — same; callable signature
  ``(train_series, horizon, **model_params) -> pd.Series``

Merge order for ``get_*``: built-ins, then entry points, then explicit
``register_*`` calls (each layer overrides the previous on name clash).
"""

from __future__ import annotations

import importlib.metadata
import warnings
from typing import Any, Callable, Dict

# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

_RECONSTRUCTION_REGISTERED: Dict[str, Callable[..., Any]] = {}
_RECONSTRUCTION_FROM_EP: Dict[str, Callable[..., Any]] = {}
_RECONSTRUCTION_EP_LOADED = False

EP_GROUP_RECONSTRUCTION = "units_missrecopred.reconstruction"
EP_GROUP_PREDICTION = "units_missrecopred.prediction"


def _iter_entry_points(group: str):
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return tuple(e for e in eps if getattr(e, "group", None) == group)


def _ensure_reconstruction_entry_points() -> None:
    global _RECONSTRUCTION_EP_LOADED, _RECONSTRUCTION_FROM_EP
    if _RECONSTRUCTION_EP_LOADED:
        return
    _RECONSTRUCTION_EP_LOADED = True
    for ep in _iter_entry_points(EP_GROUP_RECONSTRUCTION):
        try:
            fn = ep.load()
            if not callable(fn):
                warnings.warn(
                    f"Entry point {ep.name!r} in {EP_GROUP_RECONSTRUCTION!r} is not callable",
                    stacklevel=2,
                )
                continue
            _RECONSTRUCTION_FROM_EP[ep.name] = fn
        except Exception as ex:  # noqa: BLE001 — plugin load is best-effort
            warnings.warn(
                f"Failed to load entry point {ep.name!r} ({EP_GROUP_RECONSTRUCTION}): {ex}",
                stacklevel=2,
            )


def register_reconstruction_model(
    name: str,
    fn: Callable[..., Any],
    *,
    overwrite: bool = False,
) -> None:
    """Register a reconstruction callable at runtime (``name`` must be unique unless overwrite)."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    key = str(name).strip()
    if not key:
        raise ValueError("name must be non-empty")
    if key in _RECONSTRUCTION_REGISTERED and not overwrite:
        raise ValueError(f"Reconstruction model {key!r} is already registered (use overwrite=True)")
    _RECONSTRUCTION_REGISTERED[key] = fn


def get_reconstruction_models() -> Dict[str, Callable[..., Any]]:
    """Built-in dict merged with entry points and runtime registrations (registered wins)."""
    _ensure_reconstruction_entry_points()
    from reconstruction_models import RECONSTRUCTION_MODELS

    return {
        **RECONSTRUCTION_MODELS,
        **_RECONSTRUCTION_FROM_EP,
        **_RECONSTRUCTION_REGISTERED,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_PREDICTION_REGISTERED: Dict[str, Callable[..., Any]] = {}
_PREDICTION_FROM_EP: Dict[str, Callable[..., Any]] = {}
_PREDICTION_EP_LOADED = False
# Per-plugin flags (built-ins use prediction_models.GPU_MODELS / DETERMINISTIC_MODELS)
_PREDICTION_META: Dict[str, Dict[str, bool]] = {}


def _ensure_prediction_entry_points() -> None:
    global _PREDICTION_EP_LOADED, _PREDICTION_FROM_EP, _PREDICTION_META
    if _PREDICTION_EP_LOADED:
        return
    _PREDICTION_EP_LOADED = True
    for ep in _iter_entry_points(EP_GROUP_PREDICTION):
        try:
            fn = ep.load()
            if not callable(fn):
                warnings.warn(
                    f"Entry point {ep.name!r} in {EP_GROUP_PREDICTION!r} is not callable",
                    stacklevel=2,
                )
                continue
            _PREDICTION_FROM_EP[ep.name] = fn
            _PREDICTION_META.setdefault(ep.name, {"gpu": False, "deterministic": False})
        except Exception as ex:  # noqa: BLE001
            warnings.warn(
                f"Failed to load entry point {ep.name!r} ({EP_GROUP_PREDICTION}): {ex}",
                stacklevel=2,
            )


def register_prediction_model(
    name: str,
    fn: Callable[..., Any],
    *,
    gpu: bool = False,
    deterministic: bool = False,
    overwrite: bool = False,
) -> None:
    """Register a prediction callable at runtime."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    key = str(name).strip()
    if not key:
        raise ValueError("name must be non-empty")
    if key in _PREDICTION_REGISTERED and not overwrite:
        raise ValueError(f"Prediction model {key!r} is already registered (use overwrite=True)")
    _PREDICTION_REGISTERED[key] = fn
    _PREDICTION_META[key] = {"gpu": bool(gpu), "deterministic": bool(deterministic)}


def get_prediction_models() -> Dict[str, Callable[..., Any]]:
    """Built-in dict merged with entry points and runtime registrations (registered wins)."""
    _ensure_prediction_entry_points()
    from prediction_models import PREDICTION_MODELS

    return {
        **PREDICTION_MODELS,
        **_PREDICTION_FROM_EP,
        **_PREDICTION_REGISTERED,
    }


def is_prediction_plugin_gpu(name: str) -> bool:
    _ensure_prediction_entry_points()
    m = _PREDICTION_META.get(name)
    return bool(m and m.get("gpu"))


def is_prediction_plugin_deterministic(name: str) -> bool:
    _ensure_prediction_entry_points()
    m = _PREDICTION_META.get(name)
    return bool(m and m.get("deterministic"))


def clear_plugin_registry() -> None:
    """Reset runtime and entry-point caches (for tests). Call before/after tests that register plugins."""
    global _RECONSTRUCTION_EP_LOADED, _PREDICTION_EP_LOADED
    _RECONSTRUCTION_REGISTERED.clear()
    _RECONSTRUCTION_FROM_EP.clear()
    _RECONSTRUCTION_EP_LOADED = False
    _PREDICTION_REGISTERED.clear()
    _PREDICTION_FROM_EP.clear()
    _PREDICTION_META.clear()
    _PREDICTION_EP_LOADED = False

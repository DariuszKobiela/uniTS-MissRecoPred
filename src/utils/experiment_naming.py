"""Naming helpers for missingness experiments.

New files encode the two independent missingness dimensions in one filename
token (for example ``MNAR-contiguous``).  Keeping the token underscore-free
preserves compatibility with the pipeline's historical filename parsers.

Reconstructed training series use a hierarchical layout under ``fixed_dir``:

``{dataset}/{MECHANISM-structure}/{rate}p/{model}/{iteration}.csv``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


MISSINGNESS_STRUCTURES = ("scattered", "contiguous", "mixed")


def encode_missingness_label(mechanism: str, structure: str) -> str:
    mechanism = str(mechanism).strip().upper()
    structure = str(structure).strip().lower()
    if not mechanism:
        raise ValueError("Missingness mechanism must not be empty")
    if structure not in MISSINGNESS_STRUCTURES:
        raise ValueError(
            f"Unknown missingness structure {structure!r}; "
            f"expected one of {MISSINGNESS_STRUCTURES}"
        )
    return f"{mechanism}-{structure}"


def decode_missingness_label(label: str) -> tuple[str, str]:
    """Return ``(mechanism, structure)`` and accept legacy mechanism-only labels."""
    raw = str(label).strip()
    lower = raw.lower()
    for structure in MISSINGNESS_STRUCTURES:
        suffix = f"-{structure}"
        if lower.endswith(suffix):
            mechanism = raw[: -len(suffix)]
            if not mechanism:
                break
            return mechanism.upper(), structure
    # Historical MCAR/MAR/MNAR files were point/scattered realizations.
    return raw.upper(), "scattered"


def build_reconstructed_relative_path(
    dataset_name: str,
    mechanism: str,
    structure: str,
    rate_percent: int,
    iteration: int,
    model: str,
) -> Path:
    """Relative path under ``fixed_dir`` for one reconstructed realization."""
    label = encode_missingness_label(mechanism, structure)
    return Path(str(dataset_name)) / label / f"{int(rate_percent)}p" / str(model) / f"{int(iteration)}.csv"


def parse_flat_reconstructed_stem(stem: str) -> dict[str, Any]:
    """Parse legacy flat filename ``dataset_label_rateP_iter_model``."""
    parts = stem.split("_")
    rate_idx = next(
        (index for index, part in enumerate(parts) if part.endswith("p") and part[:-1].isdigit()),
        None,
    )
    if rate_idx is None or rate_idx < 1 or rate_idx + 2 >= len(parts):
        raise ValueError(f"Invalid reconstructed filename: {stem}")
    technique, structure = decode_missingness_label(parts[rate_idx - 1])
    iteration = int(parts[rate_idx + 1])
    model = "_".join(parts[rate_idx + 2 :])
    return {
        "dataset_name": "_".join(parts[: rate_idx - 1]),
        "technique": technique,
        "structure": structure,
        "rate_percent": int(parts[rate_idx][:-1]),
        "iteration": iteration,
        "model": model,
    }


def _metadata_from_hierarchical_parts(parts: tuple[str, ...]) -> dict[str, Any]:
    if len(parts) != 5:
        raise ValueError(f"Expected five path segments, got {len(parts)}: {parts!r}")
    dataset_name, label, rate_part, model, iteration_file = parts
    if not Path(iteration_file).stem.isdigit():
        raise ValueError(f"Iteration file must be numeric stem, got {iteration_file!r}")
    if not (rate_part.endswith("p") and rate_part[:-1].isdigit()):
        raise ValueError(f"Rate folder must look like 2p, got {rate_part!r}")
    technique, structure = decode_missingness_label(label)
    iteration = int(Path(iteration_file).stem)
    return {
        "dataset_name": dataset_name,
        "technique": technique,
        "structure": structure,
        "rate_percent": int(rate_part[:-1]),
        "iteration": iteration,
        "model": model,
    }


def parse_reconstructed_metadata(fixed_root: Path | str, path: Path | str) -> dict[str, Any]:
    """Parse metadata from a reconstructed CSV under ``fixed_root`` (tree or flat)."""
    fixed_root = Path(fixed_root).resolve()
    entry = Path(path).resolve()
    try:
        relative = entry.relative_to(fixed_root)
    except ValueError as exc:
        raise ValueError(f"Path {entry} is not under fixed root {fixed_root}") from exc

    parts = relative.parts
    if len(parts) == 5:
        try:
            core = _metadata_from_hierarchical_parts(parts)
        except ValueError:
            core = parse_flat_reconstructed_stem(entry.stem)
    elif len(parts) == 1:
        core = parse_flat_reconstructed_stem(entry.stem)
    else:
        raise ValueError(f"Unrecognized reconstructed path layout: {relative}")

    return {
        **core,
        "source_type": "reconstructed",
        "reconstruction_iteration": core["iteration"],
        "reconstruction_model": core["model"],
    }


def iter_reconstructed_csv_files(fixed_dir: Path | str) -> Iterator[Path]:
    """Yield every reconstructed CSV under ``fixed_dir`` (recursive)."""
    root = Path(fixed_dir)
    if not root.is_dir():
        return
    yield from sorted(root.rglob("*.csv"))

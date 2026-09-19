"""Naming helpers for missingness experiments.

New files encode the two independent missingness dimensions in one filename
token (for example ``MNAR-contiguous``).  Keeping the token underscore-free
preserves compatibility with the pipeline's historical filename parsers.
"""

from __future__ import annotations


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

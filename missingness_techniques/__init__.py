"""
Missingness Techniques Package
Contains various methods for introducing missing data into time series.
"""

from .mcar import apply_mcar
from .mar import apply_mar
from .mnar import apply_mnar

__all__ = [
    'apply_mcar',
    'apply_mar',
    'apply_mnar'
]

# Registry of all available missingness techniques
MISSINGNESS_TECHNIQUES = {
    'MCAR': apply_mcar,
    'MAR': apply_mar,
    'MNAR': apply_mnar
}


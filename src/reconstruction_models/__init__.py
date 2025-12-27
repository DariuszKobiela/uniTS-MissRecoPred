"""
Reconstruction Models Package
Contains various time series reconstruction methods.
"""

from .impute_mean import impute_mean
from .impute_median import impute_median
from .impute_mode import impute_mode
from .impute_ffill import impute_ffill
from .impute_bfill import impute_bfill
from .interpolate_nearest import interpolate_nearest
from .interpolate_linear import interpolate_linear
from .interpolate_index import interpolate_index
from .interpolate_quadratic import interpolate_quadratic
from .interpolate_cubic import interpolate_cubic
from .interpolate_polynomial import interpolate_polynomial
from .interpolate_pchip import interpolate_pchip
from .interpolate_akima import interpolate_akima
from .interpolate_spline import interpolate_spline
from .knn import knn_impute
from .sarimax import sarimax_impute
from .stable_diffusion_2_gaf import stable_diffusion_2_gaf
from .stable_diffusion_2_mtf import stable_diffusion_2_mtf
from .stable_diffusion_2_rp import stable_diffusion_2_rp
from .stable_diffusion_2_spec import stable_diffusion_2_spec

__all__ = [
    'impute_mean',
    'impute_median',
    'impute_mode',
    'impute_ffill',
    'impute_bfill',
    'interpolate_nearest',
    'interpolate_linear',
    'interpolate_index',
    'interpolate_quadratic',
    'interpolate_cubic',
    'interpolate_polynomial',
    'interpolate_pchip',
    'interpolate_akima',
    'interpolate_spline',
    'knn_impute',
    'sarimax_impute',
    'stable_diffusion_2_gaf',
    'stable_diffusion_2_mtf',
    'stable_diffusion_2_rp',
    'stable_diffusion_2_spec'
]

# Registry of all available reconstruction models
RECONSTRUCTION_MODELS = {
    'impute_mean': impute_mean,
    'impute_median': impute_median,
    'impute_mode': impute_mode,
    'impute_ffill': impute_ffill,
    'impute_bfill': impute_bfill,
    'interpolate_nearest': interpolate_nearest,
    'interpolate_linear': interpolate_linear,
    'interpolate_index': interpolate_index,
    'interpolate_quadratic': interpolate_quadratic,
    'interpolate_cubic': interpolate_cubic,
    'interpolate_polynomial': interpolate_polynomial,
    'interpolate_pchip': interpolate_pchip,
    'interpolate_akima': interpolate_akima,
    'interpolate_spline': interpolate_spline,
    'knn': knn_impute,
    'sarimax': sarimax_impute,
    'stable_diffusion_2_gaf': stable_diffusion_2_gaf,
    'stable_diffusion_2_mtf': stable_diffusion_2_mtf,
    'stable_diffusion_2_rp': stable_diffusion_2_rp,
    'stable_diffusion_2_spec': stable_diffusion_2_spec
}


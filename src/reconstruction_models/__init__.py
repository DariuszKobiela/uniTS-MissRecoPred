"""Reconstruction models used by the final experiment."""

from .impute_mean import impute_mean
from .impute_median import impute_median
from .impute_ffill import impute_ffill
from .impute_bfill import impute_bfill
from .interpolate_linear import interpolate_linear
from .interpolate_cubic import interpolate_cubic
from .interpolate_pchip import interpolate_pchip
from .knn import knn_impute
from .sarimax import sarimax_impute
from .stable_diffusion_2_gaf import stable_diffusion_2_gaf, stable_diffusion_2_gaf_finetuned
from .stable_diffusion_2_mtf import stable_diffusion_2_mtf, stable_diffusion_2_mtf_finetuned
from .stable_diffusion_2_rp import stable_diffusion_2_rp, stable_diffusion_2_rp_finetuned
from .stable_diffusion_2_spec import stable_diffusion_2_spec, stable_diffusion_2_spec_finetuned

RECONSTRUCTION_MODELS = {
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_ffill": impute_ffill,
    "impute_bfill": impute_bfill,
    "interpolate_linear": interpolate_linear,
    "interpolate_cubic": interpolate_cubic,
    "interpolate_pchip": interpolate_pchip,
    "knn": knn_impute,
    "sarimax": sarimax_impute,
    "stable_diffusion_2_gaf": stable_diffusion_2_gaf,
    "stable_diffusion_2_gaf_finetuned": stable_diffusion_2_gaf_finetuned,
    "stable_diffusion_2_mtf": stable_diffusion_2_mtf,
    "stable_diffusion_2_mtf_finetuned": stable_diffusion_2_mtf_finetuned,
    "stable_diffusion_2_rp": stable_diffusion_2_rp,
    "stable_diffusion_2_rp_finetuned": stable_diffusion_2_rp_finetuned,
    "stable_diffusion_2_spec": stable_diffusion_2_spec,
    "stable_diffusion_2_spec_finetuned": stable_diffusion_2_spec_finetuned,
}

__all__ = list(RECONSTRUCTION_MODELS)

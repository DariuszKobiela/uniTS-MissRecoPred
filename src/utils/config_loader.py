"""
Configuration Loader
Loads and manages configuration from config/config.yaml and config/prediction_models_config.yaml
"""

import copy
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


class Config:
    """Configuration manager for the framework"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to main configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_path: str = "") -> "Config":
        """
        Build Config from an in-memory mapping (e.g. after yaml.safe_load or RunConfig).

        Does not write any file; ``config_path`` is stored for logging only.
        """
        obj = cls.__new__(cls)
        obj.config_path = config_path
        obj.config = copy.deepcopy(data)
        return obj
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    # =========================================================================
    # DATA DIRECTORIES
    # =========================================================================
    
    def get_raw_source_dir(self) -> str:
        """Get raw source data directory (before cleaning)"""
        return self.config['data'].get('raw_source_dir', 'data/0_source_data')
    
    def get_cleaned_dir(self) -> str:
        """Get cleaned data directory"""
        return self.config['data'].get('cleaned_dir', 'data/1_cleaned_data')

    def get_horizon_dir(self) -> str:
        """Directory for step 1.5 horizon metadata and reports."""
        return self.config['data'].get('horizon_dir', 'data/1_5_horizon_recommendation')
    
    def get_splitted_dir(self) -> str:
        """Get splitted data base directory"""
        return self.config['data'].get('splitted_dir', 'data/2_splitted_data')
    
    def get_splitted_train_dir(self) -> str:
        """Get splitted training data directory"""
        return self.config['data'].get('splitted_train_dir', 'data/2_splitted_data/train')
    
    def get_splitted_test_dir(self) -> str:
        """Get splitted test data directory"""
        return self.config['data'].get('splitted_test_dir', 'data/2_splitted_data/test')

    def get_splitted_sd2_validation_dir(self) -> str:
        """SD2 hyperparameter validation partition (disjoint from rolling test)."""
        return self.config['data'].get(
            'splitted_sd2_validation_dir',
            'data/2_splitted_data/sd2_validation',
        )

    def get_split_manifest_path(self) -> str:
        """JSON manifest with three-way split index ranges."""
        default = str(Path(self.get_splitted_dir()) / 'split_manifest.json')
        path = self.config['data'].get('split_manifest_path')
        return str(path).strip() if path else default
    
    def get_test_samples(self) -> int:
        """Get number of samples for test set in train/test split"""
        return self.config.get('split', {}).get('test_samples', 100)

    def get_horizon_settings(self) -> Dict[str, Any]:
        """Horizon recommendation settings (split.horizons)."""
        return self.config.get('split', {}).get('horizons', {}) or {}

    def get_experiment_horizons(self) -> Dict[str, List[int]]:
        """Per-series experiment horizon lists from split.horizons.experiment."""
        raw = self.get_horizon_settings().get("experiment") or {}
        out: Dict[str, List[int]] = {}
        if not isinstance(raw, dict):
            return out
        for key, values in raw.items():
            if values is None:
                continue
            horizons = [int(v) for v in values]
            if horizons:
                out[str(key)] = sorted(set(horizons))
        return out

    def get_rolling_origin_counts(self) -> Dict[str, int]:
        """Per-series rolling-origin counts from split.horizons.rolling_origins."""
        raw = self.get_horizon_settings().get("rolling_origins") or {}
        out: Dict[str, int] = {}
        if not isinstance(raw, dict):
            return out
        for key, value in raw.items():
            if value is None:
                continue
            out[str(key)] = int(value)
        return out

    def get_seasonal_periods(self) -> Dict[str, int]:
        """Per-series seasonal period for seasonal-naive baseline."""
        raw = self.get_horizon_settings().get("seasonal_periods") or {}
        out: Dict[str, int] = {}
        if not isinstance(raw, dict):
            return out
        for key, value in raw.items():
            if value is None:
                continue
            out[str(key)] = int(value)
        return out

    def get_sd2_validation_settings(self) -> Dict[str, Any]:
        """Settings for the SD2 validation block inside split.sd2_validation."""
        return self.config.get('split', {}).get('sd2_validation', {}) or {}

    def get_rolling_origin_settings(self) -> Dict[str, Any]:
        """Rolling-origin evaluation settings from prediction.rolling_origins."""
        return self.config.get('prediction', {}).get('rolling_origins', {}) or {}

    def get_dataset_metadata_path(self) -> str:
        """Path to dataset_metadata.json written by analyze_forecast_horizons."""
        hz = self.get_horizon_settings()
        default = str(Path(self.get_horizon_dir()) / "dataset_metadata.json")
        path = hz.get('metadata_path')
        return str(path).strip() if path else default

    def get_horizon_report_csv_path(self) -> str:
        hz = self.get_horizon_settings()
        default = str(Path(self.get_horizon_dir()) / "horizon_recommendations.csv")
        path = hz.get('report_csv_path')
        return str(path).strip() if path else default

    def get_horizon_report_md_path(self) -> str:
        hz = self.get_horizon_settings()
        default = str(Path(self.get_horizon_dir()) / "horizon_recommendations.md")
        path = hz.get('report_md_path')
        return str(path).strip() if path else default
    
    def get_source_dir(self) -> str:
        """Get source data directory (training datasets for degradation)"""
        return self.config['data']['source_dir']
    
    def get_missing_dir(self) -> str:
        """Get missing data directory"""
        return self.config['data']['missing_dir']
    
    def get_fixed_dir(self) -> str:
        """Get fixed data directory"""
        return self.config['data']['fixed_dir']
    
    def get_results_dir(self) -> str:
        """Get reconstruction results directory (backward compatible)"""
        return self.get_reconstruction_results_dir()
    
    def get_reconstruction_results_dir(self) -> str:
        """Get reconstruction experiment results directory"""
        return self.config['data'].get('reconstruction_results_dir', 'reconstruction_experiments_results')
    
    def get_prediction_results_dir(self) -> str:
        """Get prediction experiment results directory"""
        return self.config['data'].get('prediction_results_dir', 'prediction_experiment_results')

    # =========================================================================
    # DATASETS
    # =========================================================================
    
    def get_datasets(self) -> List[str]:
        """
        Get list of datasets to process.
        If selected is empty, auto-discover all CSV files in source directory.
        
        Returns:
            List of dataset file paths
        """
        selected = self.config['datasets']['selected']
        
        if selected:
            # Use specified datasets
            source_dir = self.get_source_dir()
            return [os.path.join(source_dir, f) for f in selected]
        else:
            # Auto-discover all CSV files
            return self.discover_datasets()
    
    def discover_datasets(self) -> List[str]:
        """
        Auto-discover all CSV files in source directory.
        
        Returns:
            List of dataset file paths
        """
        source_dir = Path(self.get_source_dir())
        
        if not source_dir.exists():
            print(f"⚠️  Warning: Source directory not found: {source_dir}")
            return []
        
        datasets = sorted(source_dir.glob("*.csv"))
        return [str(f) for f in datasets]

    def get_dataset_aliases(self) -> Dict[str, str]:
        """Map short filename stems (degraded/reconstructed) to train stems."""
        raw = self.config.get("datasets", {}).get("aliases") or {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(k).strip(): str(v).strip()
            for k, v in raw.items()
            if str(k).strip() and str(v).strip()
        }

    def build_source_dataset_mapping(self) -> Dict[str, str]:
        """Stem → source path, including ``datasets.aliases``."""
        mapping = {Path(f).stem: f for f in self.discover_datasets()}
        for short, full in self.get_dataset_aliases().items():
            if full in mapping:
                mapping[short] = mapping[full]
        return mapping
    
    def get_csv_format(self, filename: str) -> Dict[str, Any]:
        """
        Get CSV format settings for a specific file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Dictionary with format settings (separator, decimal, index_col)
        """
        format_config = self.config['datasets']['format']
        
        # Check for special formats
        for format_name, format_settings in format_config.items():
            if format_name == 'default':
                continue
            
            if 'pattern' in format_settings:
                if format_settings['pattern'] in filename:
                    return {
                        'sep': format_settings['separator'],
                        'decimal': format_settings['decimal'],
                        'index_col': format_settings['index_col']
                    }
        
        # Use default format
        default = format_config['default']
        return {
            'sep': default['separator'],
            'decimal': default['decimal'],
            'index_col': default['index_col']
        }
    
    # =========================================================================
    # RECONSTRUCTION MODELS
    # =========================================================================
    
    def get_reconstruction_models(self) -> List[str]:
        """
        Get list of reconstruction models to use.
        If selected is empty, use all available models (excluding excluded ones).
        
        Returns:
            List of model names
        """
        from framework.plugin_registry import get_reconstruction_models

        selected = self.config['reconstruction_models'].get('selected', [])
        excluded = self.config['reconstruction_models'].get('excluded', [])
        
        # Handle None values (YAML can return None for empty lists)
        if selected is None:
            selected = []
        if excluded is None:
            excluded = []
        
        if selected:
            # Use only specified models
            return [m for m in selected if m not in excluded]
        else:
            # Use all available models except excluded (built-ins + plugins)
            all_models = list(get_reconstruction_models().keys())
            return [m for m in all_models if m not in excluded]
    
    # =========================================================================
    # PREDICTION MODELS
    # =========================================================================
    
    def get_prediction_models(self) -> List[str]:
        """
        Get list of prediction models to use.
        If selected is empty, use all available models (excluding excluded ones).
        
        Returns:
            List of model names
        """
        from framework.plugin_registry import get_prediction_models

        prediction_config = self.config.get('prediction_models', {})
        selected = prediction_config.get('selected', [])
        excluded = prediction_config.get('excluded', [])
        
        # Handle None values (YAML can return None for empty lists)
        if selected is None:
            selected = []
        if excluded is None:
            excluded = []
        
        if selected:
            # Use only specified models
            return [m for m in selected if m not in excluded]
        else:
            # Use all available models except excluded (built-ins + plugins)
            all_models = list(get_prediction_models().keys())
            return [m for m in all_models if m not in excluded]
    
    def get_predict_on_original_train(self) -> bool:
        """Get whether to predict on original training data"""
        return self.config.get('prediction', {}).get('predict_on_original_train', True)
    
    def get_predict_on_reconstructed(self) -> bool:
        """Get whether to predict on reconstructed data"""
        return self.config.get('prediction', {}).get('predict_on_reconstructed', True)

    def get_prediction_error_metrics_to_compute(self) -> List[str]:
        """
        Keys of prediction error metrics to write in script 9.
        Empty YAML list ``compute: []`` or omitted → all registered built-ins.
        """
        from prediction_metrics import list_primary_metric_keys

        em = self.config.get("prediction", {}).get("error_metrics", {})
        compute = em.get("compute")
        if compute is None:
            return list_primary_metric_keys()
        if isinstance(compute, list) and len(compute) == 0:
            return list_primary_metric_keys()
        if isinstance(compute, list):
            return [str(x).strip().lower() for x in compute]
        return [str(compute).strip().lower()]

    def get_prediction_primary_metric(self) -> str:
        """Default / primary prediction error metric (summaries, Streamlit default)."""
        return str(
            self.config.get("prediction", {}).get("error_metrics", {}).get("primary_metric", "mape")
        ).lower()

    def get_prediction_primary_metric_lower_is_better(self) -> bool:
        """
        Direction for primary_metric (e.g. ranking). Uses prediction.error_metrics.primary_metric_objective.
        """
        from prediction_metrics import infer_lower_is_better

        raw = self.config.get("prediction", {}).get("error_metrics", {}).get(
            "primary_metric_objective", "auto"
        )
        if raw is None:
            raw = "auto"
        s = str(raw).strip().lower()
        if s in ("auto", "default", "infer", ""):
            return infer_lower_is_better(self.get_prediction_primary_metric())
        if s in ("minimize", "min", "lower"):
            return True
        if s in ("maximize", "max", "higher"):
            return False
        raise ValueError(
            f"Invalid prediction.error_metrics.primary_metric_objective: {raw!r}. "
            'Use "minimize", "maximize", or "auto".'
        )

    def get_visualization_default_prediction_metric(self) -> str:
        """Streamlit default column; falls back to prediction.error_metrics.primary_metric."""
        v = self.config.get("visualization", {}).get("default_prediction_metric")
        if v is not None and str(v).strip():
            return str(v).strip().lower()
        return self.get_prediction_primary_metric()

    def get_reconstruction_error_metrics_to_compute(self) -> List[str]:
        """
        Keys of reconstruction error metrics to write in script 5.
        Empty YAML list ``compute: []`` or omitted → all registered built-ins.
        """
        from reconstruction_metrics import list_primary_metric_keys

        em = self.config.get("reconstruction", {}).get("error_metrics", {})
        compute = em.get("compute")
        if compute is None:
            return list_primary_metric_keys()
        if isinstance(compute, list) and len(compute) == 0:
            return list_primary_metric_keys()
        if isinstance(compute, list):
            return [str(x).strip().lower() for x in compute]
        return [str(compute).strip().lower()]

    def get_reconstruction_primary_metric(self) -> str:
        """Default / primary reconstruction metric (summaries in script 5, optional UI default)."""
        return str(
            self.config.get("reconstruction", {}).get("error_metrics", {}).get(
                "primary_metric", "smape"
            )
        ).lower()

    def get_reconstruction_primary_metric_lower_is_better(self) -> bool:
        """
        Direction for reconstruction primary_metric. Uses reconstruction.error_metrics.primary_metric_objective.
        """
        from reconstruction_metrics import infer_lower_is_better

        raw = self.config.get("reconstruction", {}).get("error_metrics", {}).get(
            "primary_metric_objective", "auto"
        )
        if raw is None:
            raw = "auto"
        s = str(raw).strip().lower()
        if s in ("auto", "default", "infer", ""):
            return infer_lower_is_better(self.get_reconstruction_primary_metric())
        if s in ("minimize", "min", "lower"):
            return True
        if s in ("maximize", "max", "higher"):
            return False
        raise ValueError(
            f"Invalid reconstruction.error_metrics.primary_metric_objective: {raw!r}. "
            'Use "minimize", "maximize", or "auto".'
        )
    
    # =========================================================================
    # MISSINGNESS SETTINGS
    # =========================================================================
    
    def get_missingness_techniques(self) -> List[str]:
        """
        Get list of missingness techniques to use.
        If selected is empty, use all available techniques.
        
        Returns:
            List of technique names
        """
        from missingness_techniques import MISSINGNESS_TECHNIQUES
        
        selected = self.config['missingness_techniques']['selected']
        
        if selected:
            return selected
        else:
            # Use all available techniques
            return list(MISSINGNESS_TECHNIQUES.keys())
    
    def get_missingness_rates(self) -> List[float]:
        """Get list of missingness rates"""
        return self.config['missingness_rates']['rates']

    def get_missingness_structures(self) -> List[str]:
        """Get temporal missingness structures (scattered/contiguous/mixed)."""
        raw = self.config.get("missingness_structures", {}).get("selected")
        structures = list(raw) if raw else ["scattered"]
        valid = {"scattered", "contiguous", "mixed"}
        normalized = [str(value).strip().lower() for value in structures]
        unknown = set(normalized) - valid
        if unknown:
            raise ValueError(f"Unknown missingness structures: {sorted(unknown)}")
        return normalized

    def get_missingness_structure_settings(self) -> Dict[str, Any]:
        """Parameters controlling block lengths and the mixed point share."""
        raw = self.config.get("missingness_structures", {}) or {}
        contiguous = raw.get("contiguous", {}) or {}
        mixed = raw.get("mixed", {}) or {}
        return {
            "min_block_length": int(contiguous.get("min_block_length", 3)),
            "max_block_length": int(contiguous.get("max_block_length", 24)),
            "mixed_scattered_fraction": float(mixed.get("scattered_fraction", 0.5)),
        }
    
    def get_iterations(self) -> int:
        """Get number of iterations"""
        return self.config['missingness_rates']['iterations']
    
    def get_seed(self) -> int:
        """Get random seed"""
        return self.config['missingness_rates']['seed']
    
    # =========================================================================
    # COMPUTATION SETTINGS
    # =========================================================================
    
    def get_stable_diffusion_settings(self) -> Dict[str, Any]:
        """Get Stable Diffusion model settings"""
        return self.config['computation']['stable_diffusion']
    
    def get_overwrite_existing(self) -> bool:
        """Get overwrite existing files flag (backward compatible - returns reconstruction setting)"""
        # Check new structure first
        if 'overwrite' in self.config:
            return self.config['overwrite'].get('reconstruction', False)
        # Fallback to old structure
        return self.config.get('computation', {}).get('overwrite_existing', False)
    
    def get_overwrite_reconstruction(self) -> bool:
        """Get overwrite flag for reconstruction phase (steps 3-5)"""
        return self.config.get('overwrite', {}).get('reconstruction', False)
    
    def get_overwrite_prediction(self) -> bool:
        """Get overwrite flag for prediction phase (steps 7-9)"""
        return self.config.get('overwrite', {}).get('prediction', False)
    
    def get_n_jobs(self) -> int:
        """Get number of parallel jobs"""
        return self.config['computation'].get('n_jobs', 1)
    
    def get_visualization_default_metric(self) -> str:
        """Default reconstruction metric key for Streamlit (e.g. smape, mad)."""
        return str(self.config.get("visualization", {}).get("default_metric", "smape")).lower()
    
    def get_optimization_reconstruction_metric(self) -> str:
        """Metric key used to score trials in SD hyperparameter optimization."""
        return str(self.config.get("optimization", {}).get("reconstruction_metric", "smape")).lower()

    def get_optimization_reconstruction_lower_is_better(self) -> bool:
        """
        Whether SD optimization should treat lower values of reconstruction_metric as better.

        Config key: optimization.reconstruction_metric_objective
        - "minimize", "min", "lower" -> True
        - "maximize", "max", "higher" -> False
        - "auto", null, omitted -> infer from reconstruction_metrics for the chosen metric key
        """
        from reconstruction_metrics import infer_lower_is_better

        raw = self.config.get("optimization", {}).get("reconstruction_metric_objective", "auto")
        if raw is None:
            raw = "auto"
        s = str(raw).strip().lower()
        if s in ("auto", "default", "infer", ""):
            return infer_lower_is_better(self.get_optimization_reconstruction_metric())
        if s in ("minimize", "min", "lower"):
            return True
        if s in ("maximize", "max", "higher"):
            return False
        raise ValueError(
            f"Invalid optimization.reconstruction_metric_objective: {raw!r}. "
            'Use "minimize", "maximize", or "auto".'
        )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print("\n📁 Data Directories:")
        print(f"  Raw Source:             {self.get_raw_source_dir()}")
        print(f"  Cleaned:                {self.get_cleaned_dir()}")
        print(f"  Horizon:                {self.get_horizon_dir()}")
        print(f"  Splitted:               {self.get_splitted_dir()}")
        print(f"    Train:                {self.get_splitted_train_dir()}")
        print(f"    Test:                 {self.get_splitted_test_dir()}")
        print(f"  Source:                 {self.get_source_dir()}")
        print(f"  Missing:                {self.get_missing_dir()}")
        print(f"  Fixed:                  {self.get_fixed_dir()}")
        print(f"  Reconstruction Results: {self.get_reconstruction_results_dir()}")
        print(f"  Prediction Results:     {self.get_prediction_results_dir()}")
        
        print(f"\n📊 Train/Test Split:")
        print(f"  Test samples (fallback): {self.get_test_samples()} (last N samples per dataset)")
        print(f"  Horizon dir:           {self.get_horizon_dir()}")
        print(f"  Horizon metadata:      {self.get_dataset_metadata_path()}")
        
        datasets = self.get_datasets()
        print(f"\n📊 Datasets ({len(datasets)}):")
        if datasets:
            for ds in datasets[:5]:  # Show first 5
                print(f"  - {os.path.basename(ds)}")
            if len(datasets) > 5:
                print(f"  ... and {len(datasets) - 5} more")
        else:
            print("  (none found - check source directory)")
        
        models = self.get_reconstruction_models()
        print(f"\n🔧 Reconstruction Models ({len(models)}):")
        for m in models[:10]:  # Show first 10
            print(f"  - {m}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
        
        pred_models = self.get_prediction_models()
        print(f"\n🔮 Prediction Models ({len(pred_models)}):")
        for m in pred_models[:10]:
            print(f"  - {m}")
        if len(pred_models) > 10:
            print(f"  ... and {len(pred_models) - 10} more")
        
        techniques = self.get_missingness_techniques()
        print(f"\n🎯 Missingness Techniques ({len(techniques)}):")
        for t in techniques:
            print(f"  - {t}")
        
        rates = self.get_missingness_rates()
        print(f"\n📉 Missingness Rates ({len(rates)}):")
        print(f"  {[f'{r*100:.0f}%' for r in rates]}")
        
        print(f"\n🔄 Iterations: {self.get_iterations()}")
        print(f"🎲 Random Seed: {self.get_seed()}")
        
        print(f"\n🔮 Prediction Settings:")
        print(f"  Predict on original train: {self.get_predict_on_original_train()}")
        print(f"  Predict on reconstructed:  {self.get_predict_on_reconstructed()}")

        print(f"\n📏 Reconstruction error metrics (script 5):")
        print(f"  Compute: {self.get_reconstruction_error_metrics_to_compute()}")
        print(f"  Primary: {self.get_reconstruction_primary_metric()}")
        
        print("="*70)


class PredictionModelsConfig:
    """Parameters for the active local XGBoost and SARIMAX models."""

    def __init__(self, config_path: str = "config/prediction_models_config.yaml"):
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Prediction models config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

    def get_seed(self) -> int:
        return int(self.config.get("global_training", {}).get("seed", 42))

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        return dict(self.config.get(model_name, {}))

    def get_xgboost_params(self) -> Dict[str, Any]:
        return self.get_model_params("xgboost")

    def get_sarimax_params(self) -> Dict[str, Any]:
        return self.get_model_params("sarimax")

    def print_config_summary(self) -> None:
        print("Rolling-origin model configuration")
        print(f"  seed: {self.get_seed()}")
        print(f"  xgboost: {self.get_xgboost_params()}")
        print(f"  sarimax: {self.get_sarimax_params()}")


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load main configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


def load_prediction_models_config(config_path: str = "config/prediction_models_config.yaml") -> PredictionModelsConfig:
    """
    Load prediction models configuration from file.
    
    Args:
        config_path: Path to prediction models configuration file
        
    Returns:
        PredictionModelsConfig object
    """
    return PredictionModelsConfig(config_path)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        config.print_config_summary()
        
        print("\n")
        
        pred_config = load_prediction_models_config()
        pred_config.print_config_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")

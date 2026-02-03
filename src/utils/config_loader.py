"""
Configuration Loader
Loads and manages configuration from config/config.yaml and config/prediction_models_config.yaml
"""

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
    
    def get_splitted_dir(self) -> str:
        """Get splitted data base directory"""
        return self.config['data'].get('splitted_dir', 'data/2_splitted_data')
    
    def get_splitted_train_dir(self) -> str:
        """Get splitted training data directory"""
        return self.config['data'].get('splitted_train_dir', 'data/2_splitted_data/train')
    
    def get_splitted_test_dir(self) -> str:
        """Get splitted test data directory"""
        return self.config['data'].get('splitted_test_dir', 'data/2_splitted_data/test')
    
    def get_test_samples(self) -> int:
        """Get number of samples for test set in train/test split"""
        return self.config.get('split', {}).get('test_samples', 100)
    
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
        from reconstruction_models import RECONSTRUCTION_MODELS
        
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
            # Use all available models except excluded
            all_models = list(RECONSTRUCTION_MODELS.keys())
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
        from prediction_models import PREDICTION_MODELS
        
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
            # Use all available models except excluded
            all_models = list(PREDICTION_MODELS.keys())
            return [m for m in all_models if m not in excluded]
    
    def get_predict_on_original_train(self) -> bool:
        """Get whether to predict on original training data"""
        return self.config.get('prediction', {}).get('predict_on_original_train', True)
    
    def get_predict_on_reconstructed(self) -> bool:
        """Get whether to predict on reconstructed data"""
        return self.config.get('prediction', {}).get('predict_on_reconstructed', True)
    
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
        """Get overwrite existing files flag"""
        return self.config['computation'].get('overwrite_existing', False)
    
    def get_n_jobs(self) -> int:
        """Get number of parallel jobs"""
        return self.config['computation'].get('n_jobs', 1)
    
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
        print(f"  Splitted:               {self.get_splitted_dir()}")
        print(f"    Train:                {self.get_splitted_train_dir()}")
        print(f"    Test:                 {self.get_splitted_test_dir()}")
        print(f"  Source:                 {self.get_source_dir()}")
        print(f"  Missing:                {self.get_missing_dir()}")
        print(f"  Fixed:                  {self.get_fixed_dir()}")
        print(f"  Reconstruction Results: {self.get_reconstruction_results_dir()}")
        print(f"  Prediction Results:     {self.get_prediction_results_dir()}")
        
        print(f"\n📊 Train/Test Split:")
        print(f"  Test samples:  {self.get_test_samples()} (last N samples per dataset)")
        
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
        
        print("="*70)


class PredictionModelsConfig:
    """Configuration manager for prediction models training parameters"""
    
    def __init__(self, config_path: str = "config/prediction_models_config.yaml"):
        """
        Load prediction models configuration from YAML file.
        
        Args:
            config_path: Path to prediction models configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Prediction models config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    # =========================================================================
    # GLOBAL TRAINING SETTINGS
    # =========================================================================
    
    def get_validation_split(self) -> float:
        """Get train/validation split ratio"""
        return self.config.get('global_training', {}).get('validation_split', 0.2)
    
    def get_seed(self) -> int:
        """Get random seed"""
        return self.config.get('global_training', {}).get('seed', 42)
    
    def get_max_epochs(self) -> int:
        """Get maximum training epochs"""
        return self.config.get('global_training', {}).get('max_epochs', 100)
    
    def get_batch_size(self) -> int:
        """Get training batch size"""
        return self.config.get('global_training', {}).get('batch_size', 32)
    
    def get_training_iterations(self) -> int:
        """Get number of training iterations for non-deterministic models"""
        return self.config.get('global_training', {}).get('training_iterations', 5)
    
    # =========================================================================
    # EARLY STOPPING SETTINGS
    # =========================================================================
    
    def get_early_stopping_enabled(self) -> bool:
        """Get whether early stopping is enabled"""
        return self.config.get('early_stopping', {}).get('enabled', True)
    
    def get_early_stopping_monitor(self) -> str:
        """Get metric to monitor for early stopping"""
        return self.config.get('early_stopping', {}).get('monitor', 'val_loss')
    
    def get_early_stopping_patience(self) -> int:
        """Get early stopping patience"""
        return self.config.get('early_stopping', {}).get('patience', 10)
    
    def get_early_stopping_min_delta(self) -> float:
        """Get minimum improvement delta"""
        return self.config.get('early_stopping', {}).get('min_delta', 0.001)
    
    def get_early_stopping_verbose(self) -> bool:
        """Get early stopping verbose flag"""
        return self.config.get('early_stopping', {}).get('verbose', False)
    
    # =========================================================================
    # MODEL-SPECIFIC PARAMETERS
    # =========================================================================
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.
        
        Args:
            model_name: Name of the model (lstm, gru, tcn, etc.)
            
        Returns:
            Dictionary of model parameters
        """
        return self.config.get(model_name, {})
    
    def get_lstm_params(self) -> Dict[str, Any]:
        """Get LSTM model parameters"""
        return self.config.get('lstm', {})
    
    def get_gru_params(self) -> Dict[str, Any]:
        """Get GRU model parameters"""
        return self.config.get('gru', {})
    
    def get_deepar_params(self) -> Dict[str, Any]:
        """Get DeepAR model parameters"""
        return self.config.get('deepar', {})
    
    def get_tcn_params(self) -> Dict[str, Any]:
        """Get TCN model parameters"""
        return self.config.get('tcn', {})
    
    def get_nbeats_params(self) -> Dict[str, Any]:
        """Get N-BEATS model parameters"""
        return self.config.get('nbeats', {})
    
    def get_transformer_params(self) -> Dict[str, Any]:
        """Get Transformer/TFT model parameters"""
        return self.config.get('transformer', {})
    
    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost model parameters"""
        return self.config.get('xgboost', {})
    
    def get_sarimax_params(self) -> Dict[str, Any]:
        """Get SARIMAX model parameters"""
        return self.config.get('sarimax', {})
    
    def get_holt_winters_params(self) -> Dict[str, Any]:
        """Get Holt-Winters model parameters"""
        return self.config.get('holt_winters', {})
    
    def get_prophet_params(self) -> Dict[str, Any]:
        """Get Prophet model parameters"""
        return self.config.get('prophet', {})
    
    # =========================================================================
    # MODEL CATEGORIES
    # =========================================================================
    
    def get_global_training_models(self) -> List[str]:
        """Get list of models that support global training"""
        return self.config.get('model_categories', {}).get('global_training_models', [])
    
    def get_per_file_training_models(self) -> List[str]:
        """Get list of models that require per-file training"""
        return self.config.get('model_categories', {}).get('per_file_training_models', [])
    
    def get_ml_models(self) -> List[str]:
        """Get list of machine learning models"""
        return self.config.get('model_categories', {}).get('ml_models', [])
    
    def is_global_training_model(self, model_name: str) -> bool:
        """Check if model supports global training"""
        return model_name in self.get_global_training_models()
    
    def is_per_file_training_model(self, model_name: str) -> bool:
        """Check if model requires per-file training"""
        return model_name in self.get_per_file_training_models()
    
    def get_deterministic_models(self) -> List[str]:
        """Get list of deterministic models (no need for multiple iterations)"""
        return self.config.get('model_categories', {}).get('deterministic_models', [])
    
    def get_non_deterministic_models(self) -> List[str]:
        """Get list of non-deterministic models (need multiple iterations)"""
        return self.config.get('model_categories', {}).get('non_deterministic_models', [])
    
    def is_deterministic_model(self, model_name: str) -> bool:
        """Check if model is deterministic"""
        return model_name in self.get_deterministic_models()
    
    def is_non_deterministic_model(self, model_name: str) -> bool:
        """Check if model is non-deterministic"""
        return model_name in self.get_non_deterministic_models()
    
    def get_all_model_names(self) -> List[str]:
        """
        Get list of ALL known prediction model names.
        Combines all categories: global_training + per_file + ml models.
        
        Returns:
            List of all prediction model names
        """
        all_models = set()
        all_models.update(self.get_global_training_models())
        all_models.update(self.get_per_file_training_models())
        all_models.update(self.get_ml_models())
        return list(all_models)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def print_config_summary(self):
        """Print a summary of prediction models configuration"""
        print("="*70)
        print("PREDICTION MODELS CONFIGURATION SUMMARY")
        print("="*70)
        
        print("\n⚙️ Global Training Settings:")
        print(f"  Validation split:      {self.get_validation_split()*100:.0f}%")
        print(f"  Max epochs:            {self.get_max_epochs()}")
        print(f"  Batch size:            {self.get_batch_size()}")
        print(f"  Random seed:           {self.get_seed()}")
        print(f"  Training iterations:   {self.get_training_iterations()} (for non-deterministic models)")
        
        print("\n⏱️ Early Stopping:")
        print(f"  Enabled:   {self.get_early_stopping_enabled()}")
        print(f"  Monitor:   {self.get_early_stopping_monitor()}")
        print(f"  Patience:  {self.get_early_stopping_patience()}")
        print(f"  Min delta: {self.get_early_stopping_min_delta()}")
        
        print("\n🌐 Global Training Models:")
        for m in self.get_global_training_models():
            print(f"  - {m}")
        
        print("\n📄 Per-File Training Models (statistical):")
        for m in self.get_per_file_training_models():
            print(f"  - {m}")
        
        print("\n🎲 Non-Deterministic Models (trained N times):")
        for m in self.get_non_deterministic_models():
            print(f"  - {m}")
        
        print("\n📐 Deterministic Models (trained once):")
        for m in self.get_deterministic_models():
            print(f"  - {m}")
        
        print("="*70)


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

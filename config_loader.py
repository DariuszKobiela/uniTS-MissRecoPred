"""
Configuration Loader
Loads and manages configuration from config.yaml
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """Configuration manager for the framework"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
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
    
    def get_raw_source_dir(self) -> str:
        """Get raw source data directory (before cleaning)"""
        return self.config['data'].get('raw_source_dir', 'data/0_source_data')
    
    def get_cleaned_dir(self) -> str:
        """Get cleaned data directory"""
        return self.config['data'].get('cleaned_dir', 'data/1_cleaned_data')
    
    def get_source_dir(self) -> str:
        """Get source data directory (cleaned datasets for processing)"""
        return self.config['data']['source_dir']
    
    def get_missing_dir(self) -> str:
        """Get missing data directory"""
        return self.config['data']['missing_dir']
    
    def get_fixed_dir(self) -> str:
        """Get fixed data directory"""
        return self.config['data']['fixed_dir']
    
    def get_results_dir(self) -> str:
        """Get results directory"""
        return self.config['data']['results_dir']
    
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
    
    def get_stable_diffusion_settings(self) -> Dict[str, Any]:
        """Get Stable Diffusion model settings"""
        return self.config['computation']['stable_diffusion']
    
    def get_overwrite_existing(self) -> bool:
        """Get overwrite existing files flag"""
        return self.config['computation'].get('overwrite_existing', False)
    
    def get_n_jobs(self) -> int:
        """Get number of parallel jobs"""
        return self.config['computation'].get('n_jobs', 1)
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print("\n📁 Data Directories:")
        print(f"  Source:  {self.get_source_dir()}")
        print(f"  Missing: {self.get_missing_dir()}")
        print(f"  Fixed:   {self.get_fixed_dir()}")
        print(f"  Results: {self.get_results_dir()}")
        
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
        
        techniques = self.get_missingness_techniques()
        print(f"\n🎯 Missingness Techniques ({len(techniques)}):")
        for t in techniques:
            print(f"  - {t}")
        
        rates = self.get_missingness_rates()
        print(f"\n📉 Missingness Rates ({len(rates)}):")
        print(f"  {[f'{r*100:.0f}%' for r in rates]}")
        
        print(f"\n🔄 Iterations: {self.get_iterations()}")
        print(f"🎲 Random Seed: {self.get_seed()}")
        
        print("="*70)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        config.print_config_summary()
    except Exception as e:
        print(f"Error loading configuration: {e}")


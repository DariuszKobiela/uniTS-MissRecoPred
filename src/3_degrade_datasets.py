#!/usr/bin/env python3
"""
Dataset Degradation Script
Introduces missing values into training time series datasets.
Uses config/config.yaml for configuration.

NOTE: This script operates on TRAINING data only (from data/2_splitted_data/train/).
Test data is preserved separately for prediction evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging

# Setup automatic logging to file
setup_logging("3_degrade_datasets")

from framework.plugin_registry import get_missingness_techniques
from utils.config_loader import load_config


def load_source_dataset(file_path: str, config) -> pd.DataFrame:
    """
    Load a source dataset with proper handling of different CSV formats.
    
    Args:
        file_path: Path to source CSV file
        config: Config object with format settings
        
    Returns:
        DataFrame with timestamps as index
    """
    # Get format settings for this file
    format_settings = config.get_csv_format(os.path.basename(file_path))
    
    df = pd.read_csv(
        file_path,
        **format_settings
    )
    
    return df


def process_single_degradation(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to process a single degradation task.
    
    Args:
        task: Dictionary with keys: source_file, output_file, technique, rate, seed, config, force
        
    Returns:
        Dictionary with keys: status, message, output_file
    """
    try:
        # Check if file already exists
        if Path(task['output_file']).exists() and not task['force']:
            return {
                'status': 'skipped',
                'message': f"Already exists",
                'output_file': task['output_file']
            }
        
        # Perform degradation
        degrade_dataset(
            source_file=task['source_file'],
            output_file=task['output_file'],
            missingness_technique=task['technique'],
            missing_rate=task['rate'],
            seed=task['seed'],
            config=task['config']
        )
        
        return {
            'status': 'success',
            'message': 'Completed',
            'output_file': task['output_file']
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'output_file': task['output_file']
        }


def degrade_dataset(source_file: str,
                   output_file: str,
                   missingness_technique: str,
                   missing_rate: float,
                   seed: int = None,
                   config = None) -> None:
    """
    Degrade a single dataset by introducing missing values.
    
    Args:
        source_file: Path to source CSV file
        output_file: Path to output degraded CSV file
        missingness_technique: Name of missingness technique
        missing_rate: Fraction of values to make missing (0.0 to 1.0)
        seed: Random seed for reproducibility
        config: Config object for format settings
    """
    # Check if output file already exists
    if os.path.exists(output_file) and config and not config.get_overwrite_existing():
        print(f"\n  ⏭️  Skipping (file already exists, overwrite_existing=false)")
        return
    
    # Load source dataset
    df = load_source_dataset(source_file, config)
    series = df.iloc[:, 0]  # First column is the time series
    
    all_techniques = get_missingness_techniques()
    if missingness_technique not in all_techniques:
        raise ValueError(f"Unknown missingness technique: {missingness_technique}")
    
    technique_func = all_techniques[missingness_technique]
    
    # Apply missingness
    print(f"\n  Applying {missingness_technique} with rate {missing_rate*100:.1f}%...")
    degraded_series = technique_func(series, missing_rate, seed=seed)
    
    # Create output DataFrame with original timestamps
    output_df = df.copy()
    output_df.iloc[:, 0] = degraded_series.values
    
    # Save degraded dataset
    output_df.to_csv(output_file)
    print(f"  ✓ Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Degrade univariate time series datasets by introducing missing values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config/config.yaml
  python 3_degrade_datasets.py
  
  # Override config with custom parameters
  python 3_degrade_datasets.py --techniques MCAR --rates 0.05 --iterations 3
  
  # Use custom config file
  python 3_degrade_datasets.py --config config/my_config.yaml
  
  # Specify datasets by file paths (from training split)
  python 3_degrade_datasets.py --dataset-files data/2_splitted_data/train/boiler.csv
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--dataset-files',
        nargs='+',
        help='Specific dataset files to process (overrides config)'
    )
    
    parser.add_argument(
        '--techniques',
        nargs='+',
        help='Missingness techniques to apply (overrides config)'
    )
    
    parser.add_argument(
        '--rates',
        nargs='+',
        type=float,
        help='Missing rates as fractions (overrides config)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of iterations (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Base random seed (overrides config)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing degraded datasets'
    )
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        print("   Creating default config/config.yaml...")
        return

    run_degrade_datasets(
        config,
        dataset_files=args.dataset_files,
        techniques=args.techniques,
        rates=args.rates,
        iterations=args.iterations,
        seed=args.seed,
        force=args.force,
    )


def run_degrade_datasets(
    config,
    dataset_files: List[str] | None = None,
    techniques: List[str] | None = None,
    rates: List[float] | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    force: bool = False,
) -> bool:
    """Step 3: introduce missingness in training series."""
    if dataset_files:
        ds_files = dataset_files
    else:
        ds_files = config.get_datasets()

    if not ds_files:
        print("❌ No datasets found. Check your configuration or source directory.")
        return False

    techniques = techniques if techniques else config.get_missingness_techniques()
    rates = rates if rates else config.get_missingness_rates()
    iterations = iterations if iterations is not None else config.get_iterations()
    seed = seed if seed is not None else config.get_seed()
    output_dir = config.get_missing_dir()

    os.makedirs(output_dir, exist_ok=True)

    for rate in rates:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Invalid missing rate: {rate}. Must be between 0.0 and 1.0")

    total_operations = len(ds_files) * len(techniques) * len(rates) * iterations

    print("="*70)
    print("DATASET DEGRADATION")
    print("="*70)
    print(f"Datasets: {len(ds_files)} files")
    for ds in ds_files[:5]:
        print(f"  - {os.path.basename(ds)}")
    if len(ds_files) > 5:
        print(f"  ... and {len(ds_files) - 5} more")
    print(f"Techniques: {techniques}")
    print(f"Missing rates: {[f'{r*100:.0f}%' for r in rates]}")
    print(f"Iterations: {iterations}")
    print(f"Base seed: {seed}")
    print(f"Total operations: {total_operations}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    print("\n📋 Building task list...")
    tasks = []
    for source_file in ds_files:
        if not Path(source_file).exists():
            print(f"❌ Source file not found: {source_file}")
            continue

        dataset_name = Path(source_file).stem

        for technique in techniques:
            for rate in rates:
                rate_percent = int(rate * 100)

                for iteration in range(1, iterations + 1):
                    output_filename = f"{dataset_name}_{technique}_{rate_percent}p_{iteration}.csv"
                    output_file = os.path.join(output_dir, output_filename)

                    unique_seed = seed + iteration * 1000 + ds_files.index(source_file) * 100 + len(technique) * 10 + rate_percent

                    tasks.append({
                        'source_file': source_file,
                        'output_file': output_file,
                        'technique': technique,
                        'rate': rate,
                        'seed': unique_seed,
                        'config': config,
                        'force': force
                    })

    n_jobs = config.get_n_jobs()
    print(f"🚀 Processing {len(tasks)} tasks with {n_jobs} parallel job(s)...\n")

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_degradation)(task)
        for task in tqdm(tasks, desc="⏳ Degrading datasets", unit="task", ncols=80)
    )

    completed = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    if errors > 0:
        print("\n❌ Errors occurred:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {os.path.basename(r['output_file'])}: {r['message']}")

    print("\n" + "="*70)
    print("DEGRADATION COMPLETE")
    print("="*70)
    print(f"✅ Completed: {completed}/{len(tasks)}")
    print(f"⏭️  Skipped (existing): {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Output directory: {output_dir}")
    print("="*70)
    return True


if __name__ == "__main__":
    main()


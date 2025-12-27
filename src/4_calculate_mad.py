#!/usr/bin/env python3
"""
Calculate Reconstruction Differences
Compares reconstructed datasets with original source data and calculates Mean Absolute Difference (MAD).
Results are saved to experiments_results/ with timestamp.
Uses config.yaml for configuration.
"""

import os
import sys
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import config loader
from utils.config_loader import load_config


def load_performance_metrics(results_dir: str) -> dict:
    """
    Load performance metrics from the most recent CSV file.
    Returns dict with key: (dataset_name, technique, rate, iter, model) -> metrics
    """
    perf_metrics_dir = os.path.join(results_dir, "performance_metrics")
    
    if not os.path.exists(perf_metrics_dir):
        print("⚠️  No performance metrics directory found")
        print(f"   Expected: {perf_metrics_dir}")
        print("   Run 3_reconstruct_datasets.py first to collect metrics")
        return {}
    
    # Find all performance metrics files
    perf_files = sorted(Path(perf_metrics_dir).glob("performance_metrics_*.csv"), reverse=True)
    
    if not perf_files:
        print("⚠️  No performance metrics files found")
        print(f"   Directory: {perf_metrics_dir}")
        print("   Run 3_reconstruct_datasets.py first to collect metrics")
        return {}
    
    # Use the most recent file
    latest_file = perf_files[0]
    
    try:
        df = pd.read_csv(latest_file)
        
        # Convert to dictionary with composite key
        metrics_dict = {}
        for _, row in df.iterrows():
            key = str((
                row['dataset_name'],
                row['technique'],
                row['rate_percent'],
                row['iteration'],
                row['model']
            ))
            
            metrics_dict[key] = {
                'dataset_name': row['dataset_name'],
                'technique': row['technique'],
                'rate_percent': row['rate_percent'],
                'iteration': row['iteration'],
                'model': row['model'],
                'time_seconds': row.get('time_seconds', None),
                'cpu_percent': row.get('cpu_percent', None),
                'memory_mb': row.get('memory_mb', None),
                'gpu_percent': row.get('gpu_percent', None),
                'gpu_memory_mb': row.get('gpu_memory_mb', None)
            }
        
        print(f"✅ Loaded {len(metrics_dict)} performance metric records from: {latest_file.name}")
        return metrics_dict
    except Exception as e:
        print(f"⚠️  Error loading performance metrics: {e}")
        return {}


def parse_filename(filename):
    """
    Parse reconstructed filename to extract metadata.
    Format: datasetName_technique_rateP_iteration_model.csv
    Dataset name and model name can contain underscores.
    
    Returns:
        dict with keys: dataset_name, technique, rate_percent, iteration, model
    """
    name_without_ext = filename.replace('.csv', '')
    parts = name_without_ext.split('_')
    
    if len(parts) < 5:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Find the rate pattern (XXp where XX is a number)
    rate_idx = None
    for i, part in enumerate(parts):
        if part.endswith('p') and part[:-1].isdigit():
            rate_idx = i
            break
    
    if rate_idx is None or rate_idx < 1 or rate_idx + 2 >= len(parts):
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Parse from the rate position:
    # Before rate: dataset_technique
    # Rate position: rateP
    # After rate: iteration_model...
    technique = parts[rate_idx - 1]
    rate_percent = int(parts[rate_idx].replace('p', ''))
    iteration = int(parts[rate_idx + 1])
    
    # Dataset is everything before technique
    dataset_name = '_'.join(parts[:rate_idx - 1])
    
    # Model is everything after iteration
    model = '_'.join(parts[rate_idx + 2:])
    
    return {
        'dataset_name': dataset_name,
        'technique': technique,
        'rate_percent': rate_percent,
        'iteration': iteration,
        'model': model
    }


def get_degraded_filename(dataset_name, technique, rate_percent, iteration):
    """
    Generate degraded filename from metadata.
    Format: datasetName_technique_rateP_iteration.csv
    
    Returns:
        str: Degraded filename
    """
    return f"{dataset_name}_{technique}_{rate_percent}p_{iteration}.csv"


def calculate_mad(source_file_path, degraded_file_path, reconstructed_file_path, config):
    """
    Calculate Mean Absolute Difference (MAD) between source and reconstructed data.
    ONLY for the values that were missing (destroyed) in the degraded dataset.
    
    Args:
        source_file_path: Path to source CSV file (original, complete data)
        degraded_file_path: Path to degraded CSV file (with NaN values)
        reconstructed_file_path: Path to reconstructed CSV file (repaired data)
        config: Config object for format settings
        
    Returns:
        dict with metrics: mad, max_diff, min_diff, std_diff, n_missing, n_total
    """
    # Get format settings for source file
    format_settings = config.get_csv_format(os.path.basename(source_file_path))
    
    # Load source file (original, complete data)
    source_df = pd.read_csv(source_file_path, **format_settings)
    
    # Load degraded file (with missing values)
    degraded_df = pd.read_csv(degraded_file_path, index_col=0)
    
    # Load reconstructed file (repaired data)
    reconstructed_df = pd.read_csv(reconstructed_file_path, index_col=0)
    
    # Check if all files have the same number of rows
    if not (len(source_df) == len(degraded_df) == len(reconstructed_df)):
        print(f"    ⚠️  Warning: Different number of rows - source: {len(source_df)}, degraded: {len(degraded_df)}, reconstructed: {len(reconstructed_df)}")
        min_len = min(len(source_df), len(degraded_df), len(reconstructed_df))
        source_df = source_df.head(min_len)
        degraded_df = degraded_df.head(min_len)
        reconstructed_df = reconstructed_df.head(min_len)
    
    if len(source_df) == 0:
        raise ValueError("Files are empty")
    
    # Extract values (first column after index)
    source_values = pd.to_numeric(source_df.iloc[:, 0], errors='coerce')
    degraded_values = pd.to_numeric(degraded_df.iloc[:, 0], errors='coerce')
    reconstructed_values = pd.to_numeric(reconstructed_df.iloc[:, 0], errors='coerce')
    
    # Find which values were missing in degraded dataset
    missing_mask = degraded_values.isna()
    
    if not missing_mask.any():
        raise ValueError("No missing values in degraded dataset - nothing to compare")
    
    # Extract ONLY the values that were missing
    source_missing = source_values[missing_mask]
    reconstructed_missing = reconstructed_values[missing_mask]
    
    # Remove any NaN values from comparison (shouldn't happen, but safety check)
    valid_mask = ~(source_missing.isna() | reconstructed_missing.isna())
    source_missing = source_missing[valid_mask]
    reconstructed_missing = reconstructed_missing[valid_mask]
    
    if len(source_missing) == 0:
        raise ValueError("No valid missing values to compare")
    
    # Calculate absolute differences ONLY for missing (reconstructed) values
    differences = (source_missing - reconstructed_missing).abs()
    
    # Calculate metrics
    metrics = {
        'mad': differences.mean(),           # Mean Absolute Difference (only for missing values)
        'max_diff': differences.max(),       # Maximum difference
        'min_diff': differences.min(),       # Minimum difference
        'std_diff': differences.std(),       # Standard deviation of differences
        'n_missing': len(differences),       # Number of missing values reconstructed
        'n_total': len(source_values)        # Total number of values in dataset
    }
    
    return metrics


def generate_output_filename():
    """
    Generate output filename with timestamp.
    Format: reconstruction_results_YYYYMMDD_HHMMSS.csv
    
    Returns:
        str: Output filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"reconstruction_results_{timestamp}.csv"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Calculate reconstruction differences (MAD) between source and reconstructed datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config.yaml
  python calculate_differences.py
  
  # Use custom config file
  python calculate_differences.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    # Get directories from config
    source_dir = config.get_source_dir()
    missing_dir = config.get_missing_dir()
    fixed_dir = config.get_fixed_dir()
    output_dir = config.get_results_dir()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    output_filename = generate_output_filename()
    output_file = os.path.join(output_dir, output_filename)
    
    # Auto-discover source datasets
    source_datasets = config.discover_datasets()
    if not source_datasets:
        print(f"❌ No source datasets found in {source_dir}")
        return
    
    # Create mapping: dataset_name -> source_file_path
    dataset_mapping = {Path(f).stem: f for f in source_datasets}
    
    # Get list of reconstructed files
    reconstructed_dir = Path(fixed_dir)
    if not reconstructed_dir.exists():
        print(f"❌ Fixed data directory not found: {fixed_dir}")
        return
    
    reconstructed_files = sorted(reconstructed_dir.glob("*.csv"))
    
    if not reconstructed_files:
        print(f"❌ No reconstructed datasets found in {fixed_dir}")
        return
    
    # Load performance metrics from reconstruction step
    print("\n📊 Loading performance metrics from reconstruction step...")
    performance_metrics = load_performance_metrics(output_dir)
    
    print("="*70)
    print("CALCULATE RECONSTRUCTION DIFFERENCES (MAD)")
    print("="*70)
    print(f"Source directory: {source_dir}")
    print(f"  Datasets: {len(source_datasets)} files")
    print(f"Missing data directory: {missing_dir}")
    print(f"Fixed data directory: {fixed_dir}")
    print(f"  Reconstructed: {len(reconstructed_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_filename}")
    print("="*70)
    print("\nℹ️  MAD is calculated ONLY for the values that were missing (destroyed)")
    print("="*70)
    
    # Store results
    results = []
    processed_count = 0
    error_count = 0
    
    # Process each reconstructed file
    for reconstructed_file in reconstructed_files:
        filename = reconstructed_file.name
        
        try:
            # Parse filename
            metadata = parse_filename(filename)
            
            print(f"\n[{processed_count + 1}/{len(reconstructed_files)}] Processing: {filename}")
            
            # Find corresponding source file
            dataset_name = metadata['dataset_name']
            if dataset_name not in dataset_mapping:
                print(f"  ❌ Unknown dataset: {dataset_name}")
                error_count += 1
                continue
            
            source_file_path = dataset_mapping[dataset_name]
            
            # Check if source file exists
            if not Path(source_file_path).exists():
                print(f"  ❌ Source file not found: {source_file_path}")
                error_count += 1
                continue
            
            # Find corresponding degraded file
            degraded_filename = get_degraded_filename(
                metadata['dataset_name'],
                metadata['technique'],
                metadata['rate_percent'],
                metadata['iteration']
            )
            degraded_file_path = os.path.join(missing_dir, degraded_filename)
            
            # Check if degraded file exists
            if not Path(degraded_file_path).exists():
                print(f"  ❌ Degraded file not found: {degraded_file_path}")
                error_count += 1
                continue
            
            # Calculate metrics (comparing ONLY missing values)
            metrics = calculate_mad(source_file_path, degraded_file_path, str(reconstructed_file), config)
            
            # Add result with MAD metrics
            result = {
                'dataset_name': metadata['dataset_name'],
                'technique': metadata['technique'],
                'rate_percent': metadata['rate_percent'],
                'iteration': metadata['iteration'],
                'model': metadata['model'],
                'mad': metrics['mad'],
                'max_diff': metrics['max_diff'],
                'min_diff': metrics['min_diff'],
                'std_diff': metrics['std_diff'],
                'n_missing': metrics['n_missing'],
                'n_total': metrics['n_total']
            }
            
            # Add performance metrics if available
            perf_key = str((
                metadata['dataset_name'],
                metadata['technique'],
                metadata['rate_percent'],
                metadata['iteration'],
                metadata['model']
            ))
            
            if perf_key in performance_metrics:
                perf = performance_metrics[perf_key]
                result['time_seconds'] = perf.get('time_seconds', None)
                result['cpu_percent'] = perf.get('cpu_percent', None)
                result['memory_mb'] = perf.get('memory_mb', None)
                result['gpu_percent'] = perf.get('gpu_percent', None)
                result['gpu_memory_mb'] = perf.get('gpu_memory_mb', None)
            else:
                # No performance data available (e.g., file was skipped in reconstruction)
                result['time_seconds'] = None
                result['cpu_percent'] = None
                result['memory_mb'] = None
                result['gpu_percent'] = None
                result['gpu_memory_mb'] = None
            
            results.append(result)
            
            processed_count += 1
            print(f"  ✓ MAD: {metrics['mad']:.4f}, Max Diff: {metrics['max_diff']:.4f}, Missing: {metrics['n_missing']}/{metrics['n_total']} ({metrics['n_missing']/metrics['n_total']*100:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            error_count += 1
            continue
    
    # Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("RESULTS SAVED")
        print("="*70)
        print(f"✓ Successfully processed: {processed_count}/{len(reconstructed_files)}")
        print(f"❌ Errors: {error_count}")
        print(f"📁 Results saved to: {output_file}")
        print("="*70)
        print("\nSummary Statistics:")
        print("  MAD (Mean Absolute Difference):")
        print(f"    Mean: {df_results['mad'].mean():.4f}")
        print(f"    Median: {df_results['mad'].median():.4f}")
        print(f"    Min: {df_results['mad'].min():.4f}")
        print(f"    Max: {df_results['mad'].max():.4f}")
        
        # Show performance metrics summary if available
        if 'time_seconds' in df_results.columns and df_results['time_seconds'].notna().any():
            print("\n  Performance Metrics:")
            print(f"    Total Time: {df_results['time_seconds'].sum():.2f}s")
            print(f"    Avg Time: {df_results['time_seconds'].mean():.2f}s")
            print(f"    Avg CPU: {df_results['cpu_percent'].mean():.1f}%")
            print(f"    Avg Memory: {df_results['memory_mb'].mean():.1f} MB")
            if df_results['gpu_percent'].notna().any():
                print(f"    Avg GPU: {df_results['gpu_percent'].mean():.1f}%")
        print("="*70)
    else:
        print("\n❌ No results to save.")


if __name__ == "__main__":
    main()

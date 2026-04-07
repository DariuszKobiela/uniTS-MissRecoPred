#!/usr/bin/env python3
"""
Calculate reconstruction error metrics
Compares reconstructed training data with ground truth only at positions that were
missing in the degraded series. Metrics live in the reconstruction_metrics package (one module per metric).
(extend there to add new errors; they flow to CSV, Streamlit, and SD optimization).

NOTE: This script compares reconstructed TRAINING data with original TRAINING data.
Test data is preserved separately for prediction evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import concurrent.futures

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging

# Setup automatic logging to file
setup_logging("5_calculate_reconstruction_error")

# Import config loader
from utils.config_loader import load_config
from reconstruction_metrics import (
    compute_reconstruction_metrics,
    list_primary_metric_keys,
)


def _format_recon_metric_value(key: str, value: float) -> str:
    if key == "smape":
        return f"{value:.2f}%"
    return f"{value:.4f}"


def _preview_metrics_line(metrics: dict, metric_keys: list, primary: str) -> str:
    """Short log line: primary first, then other computed keys (capped)."""
    ordered = []
    if primary in metric_keys and primary in metrics:
        ordered.append(primary)
    for k in metric_keys:
        if k != primary and k in metrics:
            ordered.append(k)
    parts = []
    for k in ordered[:5]:
        parts.append(f"{k}: {_format_recon_metric_value(k, float(metrics[k]))}")
    return ", ".join(parts)


def load_performance_metrics(results_dir: str) -> dict:
    """
    Load performance metrics from the most recent CSV file (by timestamp in filename).
    Returns dict with key: (dataset_name, technique, rate, iter, model) -> metrics
    """
    perf_metrics_dir = os.path.join(results_dir, "performance_metrics")
    
    if not os.path.exists(perf_metrics_dir):
        print("⚠️  No performance metrics directory found")
        print(f"   Expected: {perf_metrics_dir}")
        print("   Run 4_reconstruct_datasets.py first to collect metrics")
        return {}
    
    # Find all performance metrics files
    perf_files = list(Path(perf_metrics_dir).glob("performance_metrics_*.csv"))
    
    if not perf_files:
        print("⚠️  No performance metrics files found")
        print(f"   Directory: {perf_metrics_dir}")
        print("   Run 4_reconstruct_datasets.py first to collect metrics")
        return {}
    
    # Sort by timestamp in filename (YYYYMMDD_HHMMSS) - most recent first
    def extract_timestamp(filepath):
        """Extract timestamp from filename: performance_metrics_YYYYMMDD_HHMMSS.csv"""
        try:
            filename = filepath.stem  # Get filename without extension
            # Format: performance_metrics_20241227_120000
            timestamp_part = filename.replace('performance_metrics_', '')
            return timestamp_part  # Returns YYYYMMDD_HHMMSS for sorting
        except:
            return '00000000_000000'  # Fallback for invalid format
    
    perf_files_sorted = sorted(perf_files, key=extract_timestamp, reverse=True)
    
    # Use the most recent file (by timestamp)
    latest_file = perf_files_sorted[0]
    
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
                'cpu_cores_used': row.get('cpu_cores_used', None),
                'cpu_cores_total': row.get('cpu_cores_total', None),
                'memory_mb': row.get('memory_mb', None),
                'memory_total_mb': row.get('memory_total_mb', None),
                'gpu_percent': row.get('gpu_percent', None),
                'gpu_memory_mb': row.get('gpu_memory_mb', None),
                'gpu_memory_total_mb': row.get('gpu_memory_total_mb', None)
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


def calculate_reconstruction_errors(source_file_path, degraded_file_path, reconstructed_file_path, config):
    """
    Compute all registered reconstruction metrics at indices missing in the degraded file.
    """
    format_settings = config.get_csv_format(os.path.basename(source_file_path))

    source_df = pd.read_csv(source_file_path, **format_settings)
    degraded_df = pd.read_csv(degraded_file_path, index_col=0)
    reconstructed_df = pd.read_csv(reconstructed_file_path, index_col=0)

    if not (len(source_df) == len(degraded_df) == len(reconstructed_df)):
        print(f"    ⚠️  Warning: Different number of rows - source: {len(source_df)}, degraded: {len(degraded_df)}, reconstructed: {len(reconstructed_df)}")
        min_len = min(len(source_df), len(degraded_df), len(reconstructed_df))
        source_df = source_df.head(min_len)
        degraded_df = degraded_df.head(min_len)
        reconstructed_df = reconstructed_df.head(min_len)

    if len(source_df) == 0:
        raise ValueError("Files are empty")

    source_values = pd.to_numeric(source_df.iloc[:, 0], errors='coerce')
    degraded_values = pd.to_numeric(degraded_df.iloc[:, 0], errors='coerce')
    reconstructed_values = pd.to_numeric(reconstructed_df.iloc[:, 0], errors='coerce')

    missing_mask = degraded_values.isna()

    if not missing_mask.any():
        raise ValueError("No missing values in degraded dataset - nothing to compare")

    source_missing = source_values[missing_mask]
    reconstructed_missing = reconstructed_values[missing_mask]

    valid_mask = ~(source_missing.isna() | reconstructed_missing.isna())
    source_missing = source_missing[valid_mask]
    reconstructed_missing = reconstructed_missing[valid_mask]

    if len(source_missing) == 0:
        raise ValueError("No valid missing values to compare")

    metric_keys = config.get_reconstruction_error_metrics_to_compute()
    core = compute_reconstruction_metrics(
        source_missing.to_numpy(dtype=np.float64),
        reconstructed_missing.to_numpy(dtype=np.float64),
        metric_keys=metric_keys,
    )
    core["n_missing"] = int(core.pop("n_missing"))
    core["n_total"] = int(len(source_values))
    return core


def generate_output_filename():
    """
    Generate output filename with timestamp.
    Format: reconstruction_results_YYYYMMDD_HHMMSS.csv
    
    Returns:
        str: Output filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"reconstruction_results_{timestamp}.csv"


def process_file_wrapper(args):
    """
    Wrapper for processing a single file, to be used with ProcessPoolExecutor.
    Args:
        args: Tuple containing (reconstructed_file_path, dataset_mapping, missing_dir, config, performance_metrics)
    Returns:
        dict with status ('success' or 'error') and result data or error message
    """
    reconstructed_file_path, dataset_mapping, missing_dir, config, performance_metrics = args
    filename = os.path.basename(reconstructed_file_path)
    
    try:
        # Parse filename
        metadata = parse_filename(filename)
        
        # Find corresponding source file
        dataset_name = metadata['dataset_name']
        if dataset_name not in dataset_mapping:
            return {'status': 'error', 'msg': f"Unknown dataset: {dataset_name}", 'filename': filename}
        
        source_file_path = dataset_mapping[dataset_name]
        
        # Check if source file exists
        if not os.path.exists(source_file_path):
            return {'status': 'error', 'msg': f"Source file not found: {source_file_path}", 'filename': filename}
        
        # Find corresponding degraded file
        degraded_filename = get_degraded_filename(
            metadata['dataset_name'],
            metadata['technique'],
            metadata['rate_percent'],
            metadata['iteration']
        )
        degraded_file_path = os.path.join(missing_dir, degraded_filename)
        
        # Check if degraded file exists
        if not os.path.exists(degraded_file_path):
            return {'status': 'error', 'msg': f"Degraded file not found: {degraded_file_path}", 'filename': filename}
        
        metrics = calculate_reconstruction_errors(
            source_file_path, degraded_file_path, reconstructed_file_path, config
        )

        result = {
            'dataset_name': metadata['dataset_name'],
            'technique': metadata['technique'],
            'rate_percent': metadata['rate_percent'],
            'iteration': metadata['iteration'],
            'model': metadata['model'],
            **metrics,
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
            result['cpu_cores_used'] = perf.get('cpu_cores_used', None)
            result['cpu_cores_total'] = perf.get('cpu_cores_total', None)
            result['memory_mb'] = perf.get('memory_mb', None)
            result['memory_total_mb'] = perf.get('memory_total_mb', None)
            result['gpu_percent'] = perf.get('gpu_percent', None)
            result['gpu_memory_mb'] = perf.get('gpu_memory_mb', None)
            result['gpu_memory_total_mb'] = perf.get('gpu_memory_total_mb', None)
        else:
            # No performance data available
            result['time_seconds'] = None
            result['cpu_cores_used'] = None
            result['cpu_cores_total'] = None
            result['memory_mb'] = None
            result['memory_total_mb'] = None
            result['gpu_percent'] = None
            result['gpu_memory_mb'] = None
            result['gpu_memory_total_mb'] = None
            
        return {'status': 'success', 'data': result, 'filename': filename}
        
    except Exception as e:
        return {'status': 'error', 'msg': str(e), 'filename': filename}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Calculate reconstruction error metrics (MAD, MAE, RMSE, R², SMAPE, …) on missing positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config.yaml
  python calculate_differences.py
  
  # Use custom config file
  python calculate_differences.py --config config/my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return

    run_calculate_reconstruction_error(config)


def run_calculate_reconstruction_error(config) -> bool:
    """Step 5: reconstruction error metrics CSV."""
    metric_keys = config.get_reconstruction_error_metrics_to_compute()
    primary_metric = config.get_reconstruction_primary_metric()
    valid_keys = set(list_primary_metric_keys())
    unknown = [k for k in metric_keys if k not in valid_keys]
    if unknown:
        print(f"❌ Unknown reconstruction.error_metrics.compute key(s): {unknown}")
        print(f"   Valid keys: {sorted(valid_keys)}")
        return False
    if primary_metric not in metric_keys:
        print(
            f"❌ reconstruction.error_metrics.primary_metric {primary_metric!r} "
            f"must be included in compute list: {metric_keys}"
        )
        return False

    source_dir = config.get_source_dir()
    missing_dir = config.get_missing_dir()
    fixed_dir = config.get_fixed_dir()
    output_dir = config.get_results_dir()

    os.makedirs(output_dir, exist_ok=True)

    output_filename = generate_output_filename()
    output_file = os.path.join(output_dir, output_filename)

    source_datasets = config.discover_datasets()
    if not source_datasets:
        print(f"❌ No source datasets found in {source_dir}")
        return False

    dataset_mapping = {Path(f).stem: f for f in source_datasets}

    reconstructed_dir = Path(fixed_dir)
    if not reconstructed_dir.exists():
        print(f"❌ Fixed data directory not found: {fixed_dir}")
        return False

    reconstructed_files = sorted(reconstructed_dir.glob("*.csv"))

    if not reconstructed_files:
        print(f"❌ No reconstructed datasets found in {fixed_dir}")
        return False

    print("\n📊 Loading performance metrics from reconstruction step...")
    performance_metrics = load_performance_metrics(output_dir)

    print("="*70)
    print("CALCULATE RECONSTRUCTION ERROR METRICS")
    print("="*70)
    print(f"Metrics computed: {metric_keys}")
    print(f"Primary (summaries): {primary_metric}")
    print(f"Source directory: {source_dir}")
    print(f"  Datasets: {len(source_datasets)} files")
    print(f"Missing data directory: {missing_dir}")
    print(f"Fixed data directory: {fixed_dir}")
    print(f"  Reconstructed: {len(reconstructed_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_filename}")
    print("="*70)
    print("\nℹ️  Errors are computed ONLY at values that were missing (destroyed) in the degraded series")
    print("="*70)

    results = []
    processed_count = 0
    error_count = 0

    n_jobs = config.get_n_jobs()
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and len(reconstructed_files) > 1:
        print(f"\n🚀 Processing with {n_jobs} parallel jobs...")

        job_args = [
            (str(f), dataset_mapping, missing_dir, config, performance_metrics)
            for f in reconstructed_files
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_file = {executor.submit(process_file_wrapper, args): args[0] for args in job_args}

            for future in concurrent.futures.as_completed(future_to_file):
                filename = os.path.basename(future_to_file[future])
                try:
                    res = future.result()
                    processed_count += 1

                    if res['status'] == 'success':
                        metrics = res['data']
                        results.append(metrics)
                        print(f"[{processed_count}/{len(reconstructed_files)}] {filename}")
                        prev = _preview_metrics_line(metrics, metric_keys, primary_metric)
                        print(
                            f"  ✓ {prev} | Missing: {metrics['n_missing']}/{metrics['n_total']} "
                            f"({metrics['n_missing']/metrics['n_total']*100:.1f}%)"
                        )
                    else:
                        print(f"[{processed_count}/{len(reconstructed_files)}] {filename}")
                        print(f"  ❌ Error: {res['msg']}")
                        error_count += 1
                except Exception as exc:
                    print(f"[{processed_count}/{len(reconstructed_files)}] {filename}")
                    print(f"  ❌ Exception: {exc}")
                    error_count += 1
    else:
        print("\nSequential processing...")
        for reconstructed_file in reconstructed_files:
            filename = reconstructed_file.name
            args = (str(reconstructed_file), dataset_mapping, missing_dir, config, performance_metrics)

            processed_count += 1
            print(f"\n[{processed_count}/{len(reconstructed_files)}] Processing: {filename}")

            res = process_file_wrapper(args)

            if res['status'] == 'success':
                metrics = res['data']
                results.append(metrics)
                prev = _preview_metrics_line(metrics, metric_keys, primary_metric)
                print(
                    f"  ✓ {prev} | Missing: {metrics['n_missing']}/{metrics['n_total']} "
                    f"({metrics['n_missing']/metrics['n_total']*100:.1f}%)"
                )
            else:
                print(f"  ❌ Error: {res['msg']}")
                error_count += 1

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)

        print("\n" + "="*70)
        print("RESULTS SAVED")
        print("="*70)
        print(f"✓ Successfully processed: {processed_count - error_count}/{len(reconstructed_files)}")
        print(f"❌ Errors: {error_count}")
        print(f"📁 Results saved to: {output_file}")
        print("="*70)
        print("\nSummary Statistics (over all runs):")
        summary_cols = [k for k in metric_keys if k in df_results.columns]
        for col in summary_cols:
            s = df_results[col].dropna()
            if len(s) == 0:
                continue
            title = "R²" if col == "r2" else ("SMAPE (%)" if col == "smape" else col.upper())
            dec = 2 if col == "smape" else 4
            print(f"  {title}:")
            print(f"    Mean: {s.mean():.{dec}f}")
            print(f"    Median: {s.median():.{dec}f}")
            print(f"    Min: {s.min():.{dec}f}")
            print(f"    Max: {s.max():.{dec}f}")

        if 'time_seconds' in df_results.columns and df_results['time_seconds'].notna().any():
            print("\n  Performance Metrics:")
            print(f"    Total Time: {df_results['time_seconds'].sum():.2f}s")
            print(f"    Avg Time: {df_results['time_seconds'].mean():.2f}s")
            if 'cpu_cores_used' in df_results.columns and df_results['cpu_cores_used'].notna().any():
                avg_cores = df_results['cpu_cores_used'].mean()
                total_cores = df_results['cpu_cores_total'].mode()[0] if 'cpu_cores_total' in df_results.columns else 0
                print(f"    Avg CPU: {avg_cores:.2f}/{total_cores:.0f} cores")
            print(f"    Avg Memory: {df_results['memory_mb'].mean():.1f} MB")
            if df_results['gpu_percent'].notna().any():
                print(f"    Avg GPU: {df_results['gpu_percent'].mean():.1f}%")
        print("="*70)
    else:
        print("\n❌ No results to save.")

    return True


if __name__ == "__main__":
    main()

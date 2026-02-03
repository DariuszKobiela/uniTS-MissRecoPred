#!/usr/bin/env python3
"""
Calculate Prediction Error
Compares predicted values with actual test data (ground truth)
and calculates Mean Absolute Percentage Error (MAPE).
Results are saved to prediction_experiment_results/ with timestamp.
Uses config/config.yaml for configuration.

NOTE: This script compares PREDICTIONS with TEST data (ground truth).
Predictions come from 7_predict_datasets.py, test data from 2_splitted_data/test/.
"""

import os
import sys
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import concurrent.futures
import re

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import config loader
from utils.config_loader import load_config, load_prediction_models_config
from utils.logger import setup_logging

# Cache for known prediction models
_known_pred_models_cache = None

def get_known_prediction_models() -> list:
    """
    Get list of known prediction model names from config.
    Results are cached for performance.
    
    Returns:
        List of prediction model names sorted by length (longest first)
    """
    global _known_pred_models_cache
    
    if _known_pred_models_cache is None:
        try:
            pred_config = load_prediction_models_config()
            _known_pred_models_cache = pred_config.get_all_model_names()
        except Exception:
            # Fallback to hardcoded list if config fails
            _known_pred_models_cache = [
                'temporal_fusion_transformer', 'vanilla_transformer',
                'nbeats_interpretable', 'holt_winters', 'prophet', 
                'sarimax', 'xgboost', 'lstm', 'gru', 'deepar', 
                'tcn', 'nbeats', 'transformer', 'tft'
            ]
    
    # Sort by length descending (important: longer names must be matched first)
    return sorted(_known_pred_models_cache, key=len, reverse=True)

# Setup automatic logging to file
setup_logging("9_calculate_prediction_error")


def load_performance_metrics(results_dir: str) -> dict:
    """
    Load performance metrics from the most recent prediction metrics CSV file.
    Returns dict with key: (dataset_name, source_type, technique, rate, recon_iter, recon_model, pred_model, pred_iter) -> metrics
    """
    perf_metrics_dir = os.path.join(results_dir, "performance_metrics")
    
    if not os.path.exists(perf_metrics_dir):
        print("⚠️  No performance metrics directory found")
        print(f"   Expected: {perf_metrics_dir}")
        print("   Run 7_predict_datasets.py first to collect metrics")
        return {}
    
    # Find all prediction metrics files
    perf_files = list(Path(perf_metrics_dir).glob("prediction_metrics_*.csv"))
    
    if not perf_files:
        print("⚠️  No prediction metrics files found")
        print(f"   Directory: {perf_metrics_dir}")
        print("   Run 7_predict_datasets.py first to collect metrics")
        return {}
    
    # Sort by timestamp in filename (YYYYMMDD_HHMMSS) - most recent first
    def extract_timestamp(filepath):
        """Extract timestamp from filename: prediction_metrics_YYYYMMDD_HHMMSS.csv"""
        try:
            filename = filepath.stem
            timestamp_part = filename.replace('prediction_metrics_', '')
            return timestamp_part
        except:
            return '00000000_000000'
    
    perf_files_sorted = sorted(perf_files, key=extract_timestamp, reverse=True)
    latest_file = perf_files_sorted[0]
    
    try:
        df = pd.read_csv(latest_file)
        
        # Convert to dictionary with composite key
        # Normalize types to ensure matching (floats -> ints where applicable)
        metrics_dict = {}
        for _, row in df.iterrows():
            rate = row.get('rate_percent')
            recon_iter = row.get('reconstruction_iteration')
            pred_iter = row.get('prediction_iteration', 1)
            
            # Normalize numeric types (convert float to int for matching)
            rate = int(rate) if pd.notna(rate) else None
            recon_iter = int(recon_iter) if pd.notna(recon_iter) else None
            pred_iter = int(pred_iter) if pd.notna(pred_iter) else 1
            
            key = str((
                row.get('dataset_name', 'unknown'),
                row.get('source_type', 'unknown'),
                row.get('technique') if pd.notna(row.get('technique')) else None,
                rate,
                recon_iter,
                row.get('reconstruction_model') if pd.notna(row.get('reconstruction_model')) else None,
                row.get('prediction_model'),
                pred_iter
            ))
            
            metrics_dict[key] = {
                'time_seconds': row.get('time_seconds', None),
                'cpu_cores_used': row.get('cpu_cores_used', None),
                'cpu_cores_total': row.get('cpu_cores_total', None),
                'memory_mb': row.get('memory_mb', None),
                'memory_total_mb': row.get('memory_total_mb', None),
                'gpu_percent': row.get('gpu_percent', None),
                'gpu_memory_mb': row.get('gpu_memory_mb', None),
                'gpu_memory_total_mb': row.get('gpu_memory_total_mb', None),
            }
        
        print(f"✅ Loaded {len(metrics_dict)} performance metric records from: {latest_file.name}")
        return metrics_dict
    except Exception as e:
        print(f"⚠️  Error loading performance metrics: {e}")
        return {}


def parse_prediction_filename(filename: str) -> dict:
    """
    Parse prediction filename to extract metadata.
    
    Formats:
    - Original data: {dataset}_original_{pred_model}.csv (deterministic)
    - Original data: {dataset}_original_{pred_model}_iter{N}.csv (non-deterministic)
    - Reconstructed: {dataset}_{technique}_{rate}p_{iter}_{recon_model}_{pred_model}.csv (deterministic)
    - Reconstructed: {dataset}_{technique}_{rate}p_{iter}_{recon_model}_{pred_model}_iter{N}.csv (non-deterministic)
    
    Returns:
        dict with keys: dataset_name, source_type, technique, rate_percent, 
                        reconstruction_iteration, reconstruction_model, 
                        prediction_model, prediction_iteration
    """
    name_without_ext = filename.replace('.csv', '')
    
    # Check for prediction iteration suffix
    pred_iter_match = re.search(r'_iter(\d+)$', name_without_ext)
    prediction_iteration = 1
    if pred_iter_match:
        prediction_iteration = int(pred_iter_match.group(1))
        name_without_ext = name_without_ext[:pred_iter_match.start()]
    
    parts = name_without_ext.split('_')
    
    # Check if it's original data (contains "_original_")
    if '_original_' in name_without_ext:
        # Format: {dataset}_original_{pred_model}
        original_idx = parts.index('original')
        dataset_name = '_'.join(parts[:original_idx])
        prediction_model = '_'.join(parts[original_idx + 1:])
        
        return {
            'dataset_name': dataset_name,
            'source_type': 'original',
            'technique': None,
            'rate_percent': None,
            'reconstruction_iteration': None,
            'reconstruction_model': None,
            'prediction_model': prediction_model,
            'prediction_iteration': prediction_iteration
        }
    else:
        # Format: {dataset}_{technique}_{rate}p_{iter}_{recon_model}_{pred_model}
        # Find the rate pattern (XXp where XX is a number)
        rate_idx = None
        for i, part in enumerate(parts):
            if part.endswith('p') and part[:-1].isdigit():
                rate_idx = i
                break
        
        if rate_idx is None or rate_idx < 2:
            raise ValueError(f"Invalid reconstructed prediction filename format: {filename}")
        
        technique = parts[rate_idx - 1]
        rate_percent = int(parts[rate_idx].replace('p', ''))
        reconstruction_iteration = int(parts[rate_idx + 1])
        dataset_name = '_'.join(parts[:rate_idx - 1])
        
        # Everything after iteration is: recon_model_pred_model
        # We need to split this - the challenge is both can have underscores
        remaining = '_'.join(parts[rate_idx + 2:])
        
        # We need to identify where reconstruction_model ends and prediction_model starts
        # Get known prediction models from config (sorted by length, longest first)
        known_pred_models = get_known_prediction_models()
        
        # Try to find the prediction model at the end
        reconstruction_model = None
        prediction_model = None
        
        for pred_model in known_pred_models:
            if remaining.endswith(pred_model):
                pred_start = len(remaining) - len(pred_model)
                if pred_start > 0 and remaining[pred_start - 1] == '_':
                    reconstruction_model = remaining[:pred_start - 1]
                    prediction_model = pred_model
                    break
                elif pred_start == 0:
                    # whole remaining is prediction model
                    reconstruction_model = ''
                    prediction_model = pred_model
                    break
        
        if prediction_model is None:
            # Fallback: assume last part is prediction model
            remaining_parts = remaining.split('_')
            prediction_model = remaining_parts[-1]
            reconstruction_model = '_'.join(remaining_parts[:-1])
        
        return {
            'dataset_name': dataset_name,
            'source_type': 'reconstructed',
            'technique': technique,
            'rate_percent': rate_percent,
            'reconstruction_iteration': reconstruction_iteration,
            'reconstruction_model': reconstruction_model,
            'prediction_model': prediction_model,
            'prediction_iteration': prediction_iteration
        }


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Calculate Mean Absolute Percentage Error (MAPE) and other metrics.
    
    MAPE = (1/n) * Σ |actual - predicted| / |actual| * 100
    
    Args:
        actual: Series with actual (ground truth) values
        predicted: Series with predicted values
        
    Returns:
        dict with metrics: mape, mae, rmse, max_error, min_error, std_error, n_samples
    """
    # Ensure same length
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len]
        predicted = predicted.iloc[:min_len]
    
    # Convert to numeric
    actual = pd.to_numeric(actual, errors='coerce')
    predicted = pd.to_numeric(predicted, errors='coerce')
    
    # Remove NaN values
    valid_mask = ~(actual.isna() | predicted.isna())
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    
    if len(actual) == 0:
        raise ValueError("No valid values to compare")
    
    # Calculate errors
    errors = actual.values - predicted.values
    abs_errors = np.abs(errors)
    
    # Calculate MAPE (handle zero values in actual)
    # Use only non-zero actual values for MAPE calculation
    non_zero_mask = actual.values != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs(errors[non_zero_mask]) / np.abs(actual.values[non_zero_mask])) * 100
    else:
        # If all actual values are zero, MAPE is undefined
        mape = np.nan
    
    # Calculate other metrics
    mae = np.mean(abs_errors)  # Mean Absolute Error
    rmse = np.sqrt(np.mean(errors ** 2))  # Root Mean Square Error
    
    metrics = {
        'mape': mape,  # Mean Absolute Percentage Error (%)
        'mae': mae,    # Mean Absolute Error
        'rmse': rmse,  # Root Mean Square Error
        'max_error': abs_errors.max(),
        'min_error': abs_errors.min(),
        'std_error': abs_errors.std(),
        'n_samples': len(actual)
    }
    
    return metrics


def generate_output_filename():
    """
    Generate output filename with timestamp.
    Format: prediction_results_YYYYMMDD_HHMMSS.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"prediction_results_{timestamp}.csv"


def process_file_wrapper(args):
    """
    Wrapper for processing a single prediction file.
    Args:
        args: Tuple containing (prediction_file_path, test_data_mapping, config, performance_metrics)
    Returns:
        dict with status ('success' or 'error') and result data or error message
    """
    prediction_file_path, test_data_mapping, config, performance_metrics = args
    filename = os.path.basename(prediction_file_path)
    
    try:
        # Parse filename
        metadata = parse_prediction_filename(filename)
        
        # Find corresponding test file
        dataset_name = metadata['dataset_name']
        if dataset_name not in test_data_mapping:
            return {'status': 'error', 'msg': f"Unknown dataset: {dataset_name}", 'filename': filename}
        
        test_file_path = test_data_mapping[dataset_name]
        
        # Check if test file exists
        if not os.path.exists(test_file_path):
            return {'status': 'error', 'msg': f"Test file not found: {test_file_path}", 'filename': filename}
        
        # Load prediction file
        pred_df = pd.read_csv(prediction_file_path, index_col=0)
        predicted_values = pred_df['predicted'] if 'predicted' in pred_df.columns else pred_df.iloc[:, 0]
        
        # Load test file (ground truth)
        format_settings = config.get_csv_format(os.path.basename(test_file_path))
        test_df = pd.read_csv(test_file_path, **format_settings)
        actual_values = pd.to_numeric(test_df.iloc[:, 0], errors='coerce')
        
        # Calculate MAPE and other metrics
        metrics = calculate_mape(actual_values, predicted_values)
        
        # Build result
        result = {
            'dataset_name': metadata['dataset_name'],
            'source_type': metadata['source_type'],
            'technique': metadata['technique'],
            'rate_percent': metadata['rate_percent'],
            'reconstruction_iteration': metadata['reconstruction_iteration'],
            'reconstruction_model': metadata['reconstruction_model'],
            'prediction_model': metadata['prediction_model'],
            'prediction_iteration': metadata['prediction_iteration'],
            'mape': metrics['mape'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'max_error': metrics['max_error'],
            'min_error': metrics['min_error'],
            'std_error': metrics['std_error'],
            'n_samples': metrics['n_samples']
        }
        
        # Add performance metrics if available
        perf_key = str((
            metadata['dataset_name'],
            metadata['source_type'],
            metadata['technique'],
            metadata['rate_percent'],
            metadata['reconstruction_iteration'],
            metadata['reconstruction_model'],
            metadata['prediction_model'],
            metadata['prediction_iteration']
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
        description="Calculate prediction error (MAPE) between predictions and test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config/config.yaml
  python 8_calculate_prediction_error.py
  
  # Use custom config file
  python 8_calculate_prediction_error.py --config config/my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
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
    test_dir = config.get_splitted_test_dir()
    predictions_dir = os.path.join(config.get_prediction_results_dir(), "predictions")
    output_dir = config.get_prediction_results_dir()
    
    # Check directories exist
    if not os.path.exists(test_dir):
        print(f"❌ Test data directory not found: {test_dir}")
        print("   Run 2_create_split.py first to create train/test split")
        return
    
    if not os.path.exists(predictions_dir):
        print(f"❌ Predictions directory not found: {predictions_dir}")
        print("   Run 7_predict_datasets.py first to generate predictions")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    output_filename = generate_output_filename()
    output_file = os.path.join(output_dir, output_filename)
    
    # Discover test datasets
    test_files = list(Path(test_dir).glob("*.csv"))
    if not test_files:
        print(f"❌ No test datasets found in {test_dir}")
        return
    
    # Create mapping: dataset_name -> test_file_path
    test_data_mapping = {f.stem: str(f) for f in test_files}
    
    # Get list of prediction files
    prediction_files = sorted(Path(predictions_dir).glob("*.csv"))
    
    if not prediction_files:
        print(f"❌ No prediction files found in {predictions_dir}")
        return
    
    # Load performance metrics from prediction step
    print("\n📊 Loading performance metrics from prediction step...")
    performance_metrics = load_performance_metrics(output_dir)
    
    print("="*70)
    print("CALCULATE PREDICTION ERROR (MAPE)")
    print("="*70)
    print(f"Test data directory: {test_dir}")
    print(f"  Test datasets: {len(test_files)} files")
    print(f"Predictions directory: {predictions_dir}")
    print(f"  Prediction files: {len(prediction_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_filename}")
    print("="*70)
    print("\nℹ️  MAPE = Mean Absolute Percentage Error (lower is better)")
    print("="*70)
    
    # Store results
    results = []
    processed_count = 0
    error_count = 0
    
    # Determine parallel processing
    n_jobs = config.get_n_jobs()
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    # Process files
    if n_jobs > 1 and len(prediction_files) > 1:
        print(f"\n🚀 Processing with {n_jobs} parallel jobs...")
        
        # Prepare arguments for each job
        job_args = [
            (str(f), test_data_mapping, config, performance_metrics) 
            for f in prediction_files
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_file = {executor.submit(process_file_wrapper, args): args[0] for args in job_args}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                filename = os.path.basename(future_to_file[future])
                try:
                    res = future.result()
                    processed_count += 1
                    
                    if res['status'] == 'success':
                        metrics = res['data']
                        results.append(metrics)
                        mape_str = f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A"
                        print(f"[{processed_count}/{len(prediction_files)}] {filename}")
                        print(f"  ✓ MAPE: {mape_str}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
                    else:
                        print(f"[{processed_count}/{len(prediction_files)}] {filename}")
                        print(f"  ❌ Error: {res['msg']}")
                        error_count += 1
                except Exception as exc:
                    processed_count += 1
                    print(f"[{processed_count}/{len(prediction_files)}] {filename}")
                    print(f"  ❌ Exception: {exc}")
                    error_count += 1
    else:
        print("\nSequential processing...")
        for prediction_file in prediction_files:
            filename = prediction_file.name
            args = (str(prediction_file), test_data_mapping, config, performance_metrics)
            
            processed_count += 1
            print(f"\n[{processed_count}/{len(prediction_files)}] Processing: {filename}")
            
            res = process_file_wrapper(args)
            
            if res['status'] == 'success':
                metrics = res['data']
                results.append(metrics)
                mape_str = f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A"
                print(f"  ✓ MAPE: {mape_str}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
            else:
                print(f"  ❌ Error: {res['msg']}")
                error_count += 1
    
    # Save results to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("RESULTS SAVED")
        print("="*70)
        print(f"✓ Successfully processed: {processed_count - error_count}/{len(prediction_files)}")
        print(f"❌ Errors: {error_count}")
        print(f"📁 Results saved to: {output_file}")
        print("="*70)
        
        # Summary statistics
        print("\nSummary Statistics:")
        
        # MAPE statistics (excluding NaN)
        valid_mape = df_results['mape'].dropna()
        if len(valid_mape) > 0:
            print("  MAPE (Mean Absolute Percentage Error):")
            print(f"    Mean:   {valid_mape.mean():.2f}%")
            print(f"    Median: {valid_mape.median():.2f}%")
            print(f"    Min:    {valid_mape.min():.2f}%")
            print(f"    Max:    {valid_mape.max():.2f}%")
        
        # MAE statistics
        print("  MAE (Mean Absolute Error):")
        print(f"    Mean:   {df_results['mae'].mean():.4f}")
        print(f"    Median: {df_results['mae'].median():.4f}")
        print(f"    Min:    {df_results['mae'].min():.4f}")
        print(f"    Max:    {df_results['mae'].max():.4f}")
        
        # RMSE statistics
        print("  RMSE (Root Mean Square Error):")
        print(f"    Mean:   {df_results['rmse'].mean():.4f}")
        print(f"    Median: {df_results['rmse'].median():.4f}")
        print(f"    Min:    {df_results['rmse'].min():.4f}")
        print(f"    Max:    {df_results['rmse'].max():.4f}")
        
        # Breakdown by source type
        print("\n  By Source Type:")
        for source_type in df_results['source_type'].unique():
            subset = df_results[df_results['source_type'] == source_type]
            valid_mape = subset['mape'].dropna()
            if len(valid_mape) > 0:
                print(f"    {source_type}: MAPE {valid_mape.mean():.2f}% (n={len(subset)})")
        
        # Breakdown by prediction model
        print("\n  By Prediction Model:")
        for model in df_results['prediction_model'].unique():
            subset = df_results[df_results['prediction_model'] == model]
            valid_mape = subset['mape'].dropna()
            if len(valid_mape) > 0:
                print(f"    {model}: MAPE {valid_mape.mean():.2f}% (n={len(subset)})")
        
        # Performance metrics summary if available
        if 'time_seconds' in df_results.columns and df_results['time_seconds'].notna().any():
            print("\n  Performance Metrics:")
            print(f"    Total Time: {df_results['time_seconds'].sum():.2f}s")
            print(f"    Avg Time: {df_results['time_seconds'].mean():.2f}s")
        
        print("="*70)
    else:
        print("\n❌ No results to save.")


if __name__ == "__main__":
    main()

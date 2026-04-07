#!/usr/bin/env python3
"""
Calculate Prediction Error
Compares predicted values with actual test data (ground truth).
Metrics are defined in prediction_metrics/ (config: prediction.error_metrics).
Results are saved to prediction_experiment_results/ with timestamp.

NOTE: PREDICTIONS vs TEST ground truth. Predictions from 8_predict_datasets.py (after 7_train_prediction_models.py), test from splitted test dir.
"""

import os
import sys
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
from prediction_metrics import compute_prediction_metrics, get_metric_spec, list_primary_metric_keys

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
            from framework.plugin_registry import get_prediction_models

            names = set(pred_config.get_all_model_names())
            names.update(get_prediction_models().keys())
            _known_pred_models_cache = sorted(names, key=len, reverse=True)
        except Exception:
            # Fallback to hardcoded list if config fails
            try:
                from framework.plugin_registry import get_prediction_models

                _known_pred_models_cache = sorted(
                    get_prediction_models().keys(), key=len, reverse=True
                )
            except Exception:
                _known_pred_models_cache = [
                    "temporal_fusion_transformer",
                    "vanilla_transformer",
                    "nbeats_interpretable",
                    "holt_winters",
                    "prophet",
                    "sarimax",
                    "xgboost",
                    "lstm",
                    "gru",
                    "deepar",
                    "tcn",
                    "nbeats",
                    "transformer",
                    "tft",
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


def align_actual_predicted(actual: pd.Series, predicted: pd.Series) -> tuple:
    """
    Align length, coerce numeric, drop NaN pairs. Returns (y_true, y_pred) as float64 1-D arrays.
    """
    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len]
        predicted = predicted.iloc[:min_len]

    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")

    valid_mask = ~(actual.isna() | predicted.isna())
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if len(actual) == 0:
        raise ValueError("No valid values to compare")

    yt = actual.to_numpy(dtype=np.float64)
    yp = predicted.to_numpy(dtype=np.float64)
    return yt, yp


def auxiliary_error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    abs_e = np.abs(y_true - y_pred)
    return {
        "max_error": float(np.max(abs_e)),
        "min_error": float(np.min(abs_e)),
        "std_error": float(np.std(abs_e, ddof=0)),
        "n_samples": int(len(y_true)),
    }


def _format_metric_preview(key: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        spec = get_metric_spec(key)
        if spec.value_is_percent:
            return f"{float(value):.2f}%"
        return f"{float(value):.4f}"
    except KeyError:
        return f"{float(value):.4f}"


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
        args: Tuple (prediction_file_path, test_data_mapping, train_data_mapping,
              metric_keys, config, performance_metrics)
    Returns:
        dict with status ('success' or 'error') and result data or error message
    """
    prediction_file_path, test_data_mapping, train_data_mapping, metric_keys, config, performance_metrics = args
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

        y_true, y_pred = align_actual_predicted(actual_values, predicted_values)

        train_arr = None
        if dataset_name in train_data_mapping:
            train_path = train_data_mapping[dataset_name]
            if os.path.exists(train_path):
                tfmt = config.get_csv_format(os.path.basename(train_path))
                train_df = pd.read_csv(train_path, **tfmt)
                train_arr = pd.to_numeric(train_df.iloc[:, 0], errors="coerce").dropna().to_numpy(
                    dtype=np.float64
                )

        primary_metrics = compute_prediction_metrics(
            y_true, y_pred, train=train_arr, metric_keys=metric_keys
        )
        aux = auxiliary_error_stats(y_true, y_pred)

        result = {
            "dataset_name": metadata["dataset_name"],
            "source_type": metadata["source_type"],
            "technique": metadata["technique"],
            "rate_percent": metadata["rate_percent"],
            "reconstruction_iteration": metadata["reconstruction_iteration"],
            "reconstruction_model": metadata["reconstruction_model"],
            "prediction_model": metadata["prediction_model"],
            "prediction_iteration": metadata["prediction_iteration"],
            **primary_metrics,
            **aux,
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
        description="Calculate prediction error metrics (MAPE, SMAPE, MASE, MAE, RMSE, …) vs test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/9_calculate_prediction_error.py
  python src/9_calculate_prediction_error.py --config config/my_config.yaml
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

    pred_config = load_prediction_models_config()
    run_calculate_prediction_error(config, pred_config)


def run_calculate_prediction_error(config, pred_config=None) -> bool:
    """Step 9: prediction error metrics vs test split. ``pred_config`` aligns with library API (optional)."""
    _ = pred_config

    test_dir = config.get_splitted_test_dir()
    train_dir = config.get_splitted_train_dir()
    predictions_dir = os.path.join(config.get_prediction_results_dir(), "predictions")
    output_dir = config.get_prediction_results_dir()

    try:
        metric_keys = config.get_prediction_error_metrics_to_compute()
        for mk in metric_keys:
            get_metric_spec(mk)
    except (KeyError, ValueError) as e:
        print(f"❌ Invalid prediction.error_metrics.compute: {e}")
        print(f"   Known keys: {list_primary_metric_keys()}")
        return False

    primary_key = config.get_prediction_primary_metric()
    try:
        get_metric_spec(primary_key)
    except KeyError:
        print(f"❌ Unknown prediction.error_metrics.primary_metric: {primary_key!r}")
        return False

    if not os.path.exists(test_dir):
        print(f"❌ Test data directory not found: {test_dir}")
        print("   Run 2_create_split.py first to create train/test split")
        return False

    if not os.path.exists(predictions_dir):
        print(f"❌ Predictions directory not found: {predictions_dir}")
        print("   Run 7_predict_datasets.py first to generate predictions")
        return False

    os.makedirs(output_dir, exist_ok=True)

    output_filename = generate_output_filename()
    output_file = os.path.join(output_dir, output_filename)

    test_files = list(Path(test_dir).glob("*.csv"))
    if not test_files:
        print(f"❌ No test datasets found in {test_dir}")
        return False

    test_data_mapping = {f.stem: str(f) for f in test_files}

    train_data_mapping = {}
    if os.path.isdir(train_dir):
        train_files = list(Path(train_dir).glob("*.csv"))
        train_data_mapping = {f.stem: str(f) for f in train_files}
    else:
        print(f"⚠️  Train directory not found (MASE will be NaN): {train_dir}")

    prediction_files = sorted(Path(predictions_dir).glob("*.csv"))

    if not prediction_files:
        print(f"❌ No prediction files found in {predictions_dir}")
        return False

    print("\n📊 Loading performance metrics from prediction step...")
    performance_metrics = load_performance_metrics(output_dir)

    print("="*70)
    print("CALCULATE PREDICTION ERROR METRICS")
    print("="*70)
    print(f"Metrics computed: {', '.join(metric_keys)}")
    print(f"Primary (summaries): {primary_key}")
    print(f"Test data directory: {test_dir}")
    print(f"  Test datasets: {len(test_files)} files")
    print(f"Train data directory: {train_dir}")
    print(f"  Train datasets: {len(train_data_mapping)} files (for MASE scaling)")
    print(f"Predictions directory: {predictions_dir}")
    print(f"  Prediction files: {len(prediction_files)} files")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_filename}")
    print("="*70)

    results = []
    processed_count = 0
    error_count = 0

    n_jobs = config.get_n_jobs()
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and len(prediction_files) > 1:
        print(f"\n🚀 Processing with {n_jobs} parallel jobs...")

        job_args = [
            (str(f), test_data_mapping, train_data_mapping, metric_keys, config, performance_metrics)
            for f in prediction_files
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
                        pv = metrics.get(primary_key, float("nan"))
                        p_str = _format_metric_preview(primary_key, pv)
                        print(f"[{processed_count}/{len(prediction_files)}] {filename}")
                        print(f"  ✓ {primary_key.upper()}: {p_str} (n={metrics.get('n_samples', '?')})")
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
            args = (
                str(prediction_file),
                test_data_mapping,
                train_data_mapping,
                metric_keys,
                config,
                performance_metrics,
            )

            processed_count += 1
            print(f"\n[{processed_count}/{len(prediction_files)}] Processing: {filename}")

            res = process_file_wrapper(args)

            if res['status'] == 'success':
                metrics = res['data']
                results.append(metrics)
                pv = metrics.get(primary_key, float("nan"))
                p_str = _format_metric_preview(primary_key, pv)
                print(f"  ✓ {primary_key.upper()}: {p_str} (n={metrics.get('n_samples', '?')})")
            else:
                print(f"  ❌ Error: {res['msg']}")
                error_count += 1

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

        print("\nSummary Statistics:")
        for col in metric_keys:
            if col not in df_results.columns:
                continue
            s = df_results[col].dropna()
            if len(s) == 0:
                continue
            label = get_metric_spec(col).label
            pct = get_metric_spec(col).value_is_percent
            u = "%" if pct else ""
            print(f"  {label} ({col}):")
            print(f"    Mean:   {s.mean():.4f}{u}")
            print(f"    Median: {s.median():.4f}{u}")
            print(f"    Min:    {s.min():.4f}{u}")
            print(f"    Max:    {s.max():.4f}{u}")

        print("\n  By Source Type (primary metric):")
        for source_type in df_results["source_type"].unique():
            subset = df_results[df_results["source_type"] == source_type]
            if primary_key not in subset.columns:
                continue
            vp = subset[primary_key].dropna()
            if len(vp) > 0:
                print(
                    f"    {source_type}: {primary_key} "
                    f"{_format_metric_preview(primary_key, vp.mean())} (n={len(subset)})"
                )

        print("\n  By Prediction Model (primary metric):")
        for model in df_results["prediction_model"].unique():
            subset = df_results[df_results["prediction_model"] == model]
            if primary_key not in subset.columns:
                continue
            vp = subset[primary_key].dropna()
            if len(vp) > 0:
                print(
                    f"    {model}: {primary_key} "
                    f"{_format_metric_preview(primary_key, vp.mean())} (n={len(subset)})"
                )

        if 'time_seconds' in df_results.columns and df_results['time_seconds'].notna().any():
            print("\n  Performance Metrics:")
            print(f"    Total Time: {df_results['time_seconds'].sum():.2f}s")
            print(f"    Avg Time: {df_results['time_seconds'].mean():.2f}s")

        print("="*70)
    else:
        print("\n❌ No results to save.")

    return True


if __name__ == "__main__":
    main()

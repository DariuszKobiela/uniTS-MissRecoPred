#!/usr/bin/env python3
"""
Dataset Prediction Script
Predicts future values (test set) using training data.

This script:
1. Loads training data (original or reconstructed)
2. Trains prediction models on the training data
3. Predicts future values (horizon = test set size)
4. Saves predictions and performance metrics

Uses config.yaml for configuration.
Collects performance metrics (time, CPU, RAM, GPU usage).

NOTE: This script predicts on TRAINING data to forecast TEST data values.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import prediction models and config loader
from prediction_models import PREDICTION_MODELS, is_gpu_model
from utils.config_loader import load_config
from utils.performance_metrics import PerformanceMonitor, format_metrics


def load_training_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a training dataset.
    
    Args:
        file_path: Path to training CSV file
        
    Returns:
        DataFrame with index and values
    """
    df = pd.read_csv(file_path, index_col=0)
    
    # Convert first column to numeric
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    # Try to parse index as datetime or numeric
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            pass
    
    return df


def load_test_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a test dataset (ground truth for evaluation).
    
    Args:
        file_path: Path to test CSV file
        
    Returns:
        DataFrame with index and values
    """
    return load_training_dataset(file_path)


def process_single_prediction(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to process a single prediction task.
    
    Args:
        task: Dictionary with keys: train_file, output_file, model_name, horizon, config, etc.
        
    Returns:
        Dictionary with keys: status, message, output_file, model, metrics, metadata, predictions
    """
    try:
        # Check if file already exists
        if Path(task['output_file']).exists() and not task['force']:
            return {
                'status': 'skipped',
                'message': 'Already exists',
                'output_file': task['output_file'],
                'model': task['model_name'],
                'metadata': task.get('metadata', {}),
                'metrics': None,
                'predictions': None
            }
        
        # Perform prediction and collect metrics
        predictions, metrics = predict_dataset(
            train_file=task['train_file'],
            output_file=task['output_file'],
            prediction_model=task['model_name'],
            horizon=task['horizon'],
            config=task['config'],
            seed=task.get('seed')
        )
        
        return {
            'status': 'success',
            'message': 'Completed',
            'output_file': task['output_file'],
            'model': task['model_name'],
            'metadata': task.get('metadata', {}),
            'metrics': metrics,
            'predictions': predictions
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'output_file': task['output_file'],
            'model': task['model_name'],
            'metadata': task.get('metadata', {}),
            'metrics': None,
            'predictions': None
        }


def predict_dataset(train_file: str,
                    output_file: str,
                    prediction_model: str,
                    horizon: int,
                    config=None,
                    seed: int = None) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Predict future values for a single dataset.
    
    Args:
        train_file: Path to training CSV file
        output_file: Path to output prediction CSV file
        prediction_model: Name of prediction model to use
        horizon: Number of steps to predict
        config: Configuration object
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (predictions Series, metrics dict)
    """
    # Load training dataset
    df = load_training_dataset(train_file)
    series = df.iloc[:, 0]  # First column is the time series
    
    # Get prediction model function
    if prediction_model not in PREDICTION_MODELS:
        raise ValueError(f"Unknown prediction model: {prediction_model}")
    
    model_func = PREDICTION_MODELS[prediction_model]
    
    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Apply prediction
    print(f"    Applying {prediction_model}...")
    predictions = model_func(series, horizon, random_state=seed)
    
    # Stop monitoring and collect metrics
    metrics = monitor.stop()
    
    # Create output DataFrame with predictions
    output_df = pd.DataFrame({
        'predicted': predictions.values
    }, index=predictions.index)
    output_df.index.name = 'index'
    
    # Save predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file)
    print(f"    ✓ Saved to: {output_file}")
    print(f"    📊 {format_metrics(metrics)}")
    
    return predictions, metrics


def parse_reconstructed_filename(filename: str) -> dict:
    """
    Parse reconstructed filename to extract metadata.
    Format: datasetName_technique_rateP_iteration_model.csv
    
    Returns:
        dict with keys: dataset, technique, rate_percent, iteration, reconstruction_model
    """
    base_name = filename.replace('.csv', '')
    parts = base_name.split('_')
    
    if len(parts) < 5:
        raise ValueError(f"Invalid reconstructed filename format: {filename}")
    
    # Find the rate pattern (XXp where XX is a number)
    rate_idx = None
    for i, part in enumerate(parts):
        if part.endswith('p') and part[:-1].isdigit():
            rate_idx = i
            break
    
    if rate_idx is None or rate_idx < 1 or rate_idx + 2 >= len(parts):
        raise ValueError(f"Invalid filename format: {filename}")
    
    technique = parts[rate_idx - 1]
    rate_percent = int(parts[rate_idx].replace('p', ''))
    iteration = int(parts[rate_idx + 1])
    dataset = '_'.join(parts[:rate_idx - 1])
    reconstruction_model = '_'.join(parts[rate_idx + 2:])
    
    return {
        'dataset': dataset,
        'technique': technique,
        'rate_percent': rate_percent,
        'iteration': iteration,
        'reconstruction_model': reconstruction_model
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict future values using training data with various prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config.yaml
  python 7_predict_datasets.py
  
  # Override with specific models
  python 7_predict_datasets.py --models holt_winters prophet lstm
  
  # Predict only on reconstructed data
  python 7_predict_datasets.py --no-original
  
  # Use custom config file
  python 7_predict_datasets.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(PREDICTION_MODELS.keys()) + ['all'],
        help='Prediction models to apply (overrides config)'
    )
    
    parser.add_argument(
        '--no-original',
        action='store_true',
        help='Do not predict on original training data (only reconstructed)'
    )
    
    parser.add_argument(
        '--no-reconstructed',
        action='store_true',
        help='Do not predict on reconstructed data (only original)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing prediction files'
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
    train_dir = config.get_splitted_train_dir()
    test_dir = config.get_splitted_test_dir()
    fixed_dir = config.get_fixed_dir()
    output_dir = config.get_prediction_results_dir()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Get prediction settings
    predict_on_original = not args.no_original and config.get_predict_on_original_train()
    predict_on_reconstructed = not args.no_reconstructed and config.get_predict_on_reconstructed()
    
    # Get models - priority: CLI args > config
    if args.models:
        if 'all' in args.models:
            models = list(PREDICTION_MODELS.keys())
        else:
            models = args.models
    else:
        models = config.get_prediction_models()
    
    if not models:
        print("❌ No prediction models specified")
        return
    
    # Get horizon (number of steps to predict) from test data
    test_files = list(Path(test_dir).glob("*.csv"))
    if not test_files:
        print(f"❌ No test data found in {test_dir}")
        print("   Run 2_create_split.py first to create train/test split")
        return
    
    # Determine horizon from first test file
    test_df = load_test_dataset(str(test_files[0]))
    horizon = len(test_df)
    print(f"📊 Prediction horizon: {horizon} steps (from test set size)")
    
    # Get seed
    seed = config.get_seed()
    overwrite = args.force if args.force else config.get_overwrite_existing()
    
    # Build task list
    tasks = []
    
    # Task 1: Predictions on original training data
    if predict_on_original:
        print(f"\n📋 Building tasks for original training data...")
        train_files = list(Path(train_dir).glob("*.csv"))
        
        for train_file in train_files:
            dataset_name = train_file.stem
            
            for model_name in models:
                output_filename = f"{dataset_name}_original_{model_name}.csv"
                output_file = os.path.join(predictions_dir, output_filename)
                
                tasks.append({
                    'train_file': str(train_file),
                    'output_file': output_file,
                    'model_name': model_name,
                    'horizon': horizon,
                    'config': config,
                    'seed': seed,
                    'force': overwrite,
                    'metadata': {
                        'dataset': dataset_name,
                        'source_type': 'original',
                        'technique': None,
                        'rate_percent': None,
                        'iteration': None,
                        'reconstruction_model': None
                    }
                })
    
    # Task 2: Predictions on reconstructed data
    if predict_on_reconstructed:
        print(f"\n📋 Building tasks for reconstructed data...")
        reconstructed_files = list(Path(fixed_dir).glob("*.csv"))
        
        if not reconstructed_files:
            print(f"⚠️  No reconstructed files found in {fixed_dir}")
            print("   Run steps 3-4 first to create reconstructed data")
        else:
            for recon_file in reconstructed_files:
                try:
                    metadata = parse_reconstructed_filename(recon_file.name)
                    
                    for model_name in models:
                        # Output filename includes reconstruction info
                        base_name = recon_file.stem
                        output_filename = f"{base_name}_{model_name}.csv"
                        output_file = os.path.join(predictions_dir, output_filename)
                        
                        tasks.append({
                            'train_file': str(recon_file),
                            'output_file': output_file,
                            'model_name': model_name,
                            'horizon': horizon,
                            'config': config,
                            'seed': seed,
                            'force': overwrite,
                            'metadata': {
                                'dataset': metadata['dataset'],
                                'source_type': 'reconstructed',
                                'technique': metadata['technique'],
                                'rate_percent': metadata['rate_percent'],
                                'iteration': metadata['iteration'],
                                'reconstruction_model': metadata['reconstruction_model']
                            }
                        })
                except Exception as e:
                    print(f"⚠️  Warning: Could not parse filename {recon_file.name}: {e}")
                    continue
    
    if not tasks:
        print("❌ No prediction tasks to run")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET PREDICTION")
    print("="*70)
    print(f"Prediction models ({len(models)}): {models[:5]}{'...' if len(models) > 5 else ''}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Horizon: {horizon} steps")
    print(f"Output directory: {output_dir}")
    print(f"Predict on original training: {predict_on_original}")
    print(f"Predict on reconstructed: {predict_on_reconstructed}")
    print("="*70)
    
    # Separate GPU and CPU tasks
    gpu_tasks = [t for t in tasks if is_gpu_model(t['model_name'])]
    cpu_tasks = [t for t in tasks if not is_gpu_model(t['model_name'])]
    
    n_jobs = config.get_n_jobs()
    print(f"\n🚀 Processing {len(tasks)} tasks:")
    print(f"   - {len(cpu_tasks)} CPU model tasks (parallel with {n_jobs} jobs)")
    print(f"   - {len(gpu_tasks)} GPU model tasks (sequential)\n")
    
    all_results = []
    
    # Process CPU models in parallel
    if cpu_tasks:
        print(f"⚡ Processing CPU models in parallel...")
        cpu_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_single_prediction)(task)
            for task in tqdm(cpu_tasks, desc="⏳ CPU models", unit="task", ncols=80)
        )
        all_results.extend(cpu_results)
    
    # Process GPU models sequentially
    if gpu_tasks:
        print(f"\n🎨 Processing GPU models sequentially...")
        gpu_results = Parallel(n_jobs=1, backend='loky')(
            delayed(process_single_prediction)(task)
            for task in tqdm(gpu_tasks, desc="⏳ GPU models", unit="task", ncols=80)
        )
        all_results.extend(gpu_results)
    
    # Count results
    completed = sum(1 for r in all_results if r['status'] == 'success')
    skipped = sum(1 for r in all_results if r['status'] == 'skipped')
    errors = sum(1 for r in all_results if r['status'] == 'error')
    
    # Print errors if any
    if errors > 0:
        print("\n❌ Errors occurred:")
        error_results = [r for r in all_results if r['status'] == 'error']
        for r in error_results[:10]:
            print(f"  - {r['model']}: {r['message']}")
        if len(error_results) > 10:
            print(f"  ... and {len(error_results) - 10} more errors")
    
    # Save performance metrics
    print("\n💾 Saving performance metrics...")
    perf_metrics_dir = os.path.join(output_dir, "performance_metrics")
    os.makedirs(perf_metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(perf_metrics_dir, f"prediction_metrics_{timestamp}.csv")
    
    # Collect metrics from all successful predictions
    metrics_data = []
    for result in all_results:
        if result['status'] == 'success' and result.get('metrics'):
            metadata = result.get('metadata', {})
            metrics = result['metrics']
            
            metrics_data.append({
                'dataset_name': metadata.get('dataset', 'unknown'),
                'source_type': metadata.get('source_type', 'unknown'),
                'technique': metadata.get('technique'),
                'rate_percent': metadata.get('rate_percent'),
                'iteration': metadata.get('iteration'),
                'reconstruction_model': metadata.get('reconstruction_model'),
                'prediction_model': result['model'],
                'time_seconds': metrics.get('time_seconds', 0),
                'cpu_cores_used': metrics.get('cpu_cores_used', 0),
                'cpu_cores_total': metrics.get('cpu_cores_total', 0),
                'memory_mb': metrics.get('memory_mb', 0),
                'memory_total_mb': metrics.get('memory_total_mb', 0),
                'gpu_percent': metrics.get('gpu_percent', None),
                'gpu_memory_mb': metrics.get('gpu_memory_mb', None),
                'gpu_memory_total_mb': metrics.get('gpu_memory_total_mb', None),
                'timestamp': timestamp
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"   ✓ Saved {len(metrics_data)} performance records to: {metrics_file}")
    else:
        print("   ⚠️  No performance metrics collected")
    
    # Summary
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"✅ Completed: {completed}/{len(tasks)}")
    print(f"⏭️  Skipped (existing): {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Predictions directory: {predictions_dir}")
    if metrics_data:
        print(f"📊 Performance metrics: {metrics_file}")
    print("="*70)


if __name__ == "__main__":
    main()

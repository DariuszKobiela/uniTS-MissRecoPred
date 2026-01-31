#!/usr/bin/env python3
"""
Dataset Prediction Script
Predicts future values (test set) using trained models.

This script:
1. Loads pre-trained models from trained_models/ folder
2. Performs predictions on all training files
3. For statistical models (SARIMAX, Holt-Winters, Prophet) - trains per-file and predicts

Uses config/config.yaml and config/prediction_models_config.yaml for configuration.
Collects prediction performance metrics (time, CPU, RAM).
"""

# Suppress PyTorch Lightning output BEFORE any imports
import os
os.environ["PL_TRAINER_GPUS"] = "0"  # Suppress GPU detection logs
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow if used

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.fabric").setLevel(logging.CRITICAL)

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import torch
import warnings
import re

# Suppress noisy warnings from libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pytorch_lightning.*')
warnings.filterwarnings('ignore', message='.*predict_dataloader.*')
warnings.filterwarnings('ignore', message='.*num_workers.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*Maximum Likelihood.*')
warnings.filterwarnings('ignore', message='.*optimization failed.*')

# Optimize for GPUs with Tensor Cores
torch.set_float32_matmul_precision('medium')

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import config loader and prediction models
from utils.config_loader import load_config, load_prediction_models_config
from utils.performance_metrics import PerformanceMonitor
from utils.logger import setup_logging, EpochLogger

# Setup automatic logging to file
setup_logging("8_predict_datasets")


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from CSV file."""
    df = pd.read_csv(file_path, index_col=0)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            pass
    
    return df


def series_to_darts(series: pd.Series):
    """Convert pandas Series to Darts TimeSeries."""
    from darts import TimeSeries
    
    date_index = pd.date_range(start='2000-01-01', periods=len(series), freq='h')
    ts = TimeSeries.from_times_and_values(times=date_index, values=series.values, freq='h')
    return ts


def parse_reconstructed_filename(filename: str) -> dict:
    """Parse reconstructed filename to extract metadata."""
    base_name = filename.replace('.csv', '')
    parts = base_name.split('_')
    
    if len(parts) < 5:
        raise ValueError(f"Invalid reconstructed filename format: {filename}")
    
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


import contextlib
import io

@contextlib.contextmanager
def suppress_all_output():
    """Suppress all stdout and stderr output at fd level."""
    # Get the real file descriptors (handle TeeOutput wrapper)
    real_stdout = getattr(sys.stdout, 'original', sys.stdout)
    real_stderr = getattr(sys.stderr, 'original', sys.stderr)
    
    # Get file descriptor numbers
    try:
        stdout_fd = real_stdout.fileno()
        stderr_fd = real_stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # Fallback if fileno not available
        yield
        return
    
    # Save copies of the original fds
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        # Open /dev/null and redirect
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore original fds
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def load_trained_model(model_name: str, iteration: int, models_dir: str):
    """
    Load a trained model from file.
    
    Returns:
        Loaded model (Darts model or dict with XGBoost model and lag)
    """
    # Try Darts model first (.pt)
    darts_path = os.path.join(models_dir, f"{model_name}_iter{iteration}.pt")
    if os.path.exists(darts_path):
        # Import appropriate model class
        from darts.models import RNNModel, TCNModel, NBEATSModel, TFTModel, TransformerModel
        
        # Suppress PyTorch Lightning output during model loading
        with suppress_all_output():
            if model_name in ['lstm', 'gru', 'deepar']:
                model = RNNModel.load(darts_path)
            elif model_name == 'tcn':
                model = TCNModel.load(darts_path)
            elif model_name == 'nbeats':
                model = NBEATSModel.load(darts_path)
            elif model_name == 'vanilla_transformer':
                model = TransformerModel.load(darts_path)
            elif model_name == 'temporal_fusion_transformer':
                model = TFTModel.load(darts_path)
            else:
                raise ValueError(f"Unknown Darts model type: {model_name}")
        
        return model
    
    # Try pickle model (.pkl)
    pkl_path = os.path.join(models_dir, f"{model_name}_iter{iteration}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    raise FileNotFoundError(f"Model not found: {model_name}_iter{iteration} in {models_dir}")


def get_available_trained_models(models_dir: str) -> Dict[str, List[int]]:
    """
    Get available trained models and their iterations.
    
    Returns:
        Dict mapping model_name -> list of available iterations
    """
    available = {}
    
    if not os.path.exists(models_dir):
        return available
    
    for file in Path(models_dir).glob("*_iter*.pt"):
        match = re.match(r'(.+)_iter(\d+)\.pt', file.name)
        if match:
            model_name = match.group(1)
            iteration = int(match.group(2))
            if model_name not in available:
                available[model_name] = []
            available[model_name].append(iteration)
    
    for file in Path(models_dir).glob("*_iter*.pkl"):
        match = re.match(r'(.+)_iter(\d+)\.pkl', file.name)
        if match:
            model_name = match.group(1)
            iteration = int(match.group(2))
            if model_name not in available:
                available[model_name] = []
            available[model_name].append(iteration)
    
    # Sort iterations
    for model_name in available:
        available[model_name] = sorted(available[model_name])
    
    return available


def predict_with_darts_model(model, train_series: pd.Series, horizon: int) -> np.ndarray:
    """Make predictions with a Darts model."""
    ts = series_to_darts(train_series)
    
    predictions = []
    current_ts = ts
    
    for _ in range(horizon):
        with suppress_all_output():
            pred = model.predict(n=1, series=current_ts, verbose=False)
        pred_value = pred.values()[0, 0]
        predictions.append(pred_value)
        
        # Extend series with prediction
        new_date = current_ts.end_time() + pd.Timedelta(hours=1)
        new_ts = pd.Series([pred_value], index=[new_date])
        current_ts = current_ts.append_values(new_ts.values)
    
    return np.array(predictions)


def predict_with_xgboost(model_data: dict, train_series: pd.Series, horizon: int) -> np.ndarray:
    """Make predictions with XGBoost model using recursive forecasting."""
    model = model_data['model']
    lag = model_data['lag']
    
    values = train_series.values.tolist()
    predictions = []
    
    for _ in range(horizon):
        if len(values) < lag:
            # Not enough history, use simple fallback
            predictions.append(values[-1] if values else 0)
        else:
            X = np.array([values[-lag:]])
            pred = model.predict(X)[0]
            predictions.append(pred)
            values.append(pred)
    
    return np.array(predictions)


def predict_with_statistical_model(model_name: str, train_series: pd.Series, 
                                    horizon: int, pred_config) -> np.ndarray:
    """Train and predict with statistical models (per-file)."""
    from prediction_models import PREDICTION_MODELS
    
    predict_func = PREDICTION_MODELS.get(model_name)
    if predict_func is None:
        raise ValueError(f"Unknown statistical model: {model_name}")
    
    # Get model parameters
    model_params = pred_config.get_model_params(model_name)
    
    # Statistical models predict function returns predictions directly
    predictions = predict_func(train_series, horizon, **model_params)
    
    return np.array(predictions)


def save_predictions(predictions: np.ndarray, output_path: str, 
                     train_series: pd.Series, iteration: int = 1):
    """Save predictions to CSV file."""
    # Create index for predictions (continuing from training data)
    if hasattr(train_series.index, 'freq') and train_series.index.freq:
        pred_index = pd.date_range(
            start=train_series.index[-1] + train_series.index.freq,
            periods=len(predictions),
            freq=train_series.index.freq
        )
    else:
        pred_index = range(len(train_series), len(train_series) + len(predictions))
    
    df = pd.DataFrame({
        'predicted': predictions,
        'iteration': iteration
    }, index=pred_index)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)


def process_single_file_statistical(args):
    """
    Worker function for parallel processing of statistical models.
    Returns metrics dict or None on error.
    """
    # Suppress warnings in worker process
    import warnings
    warnings.filterwarnings('ignore')
    
    file_info, model_name, horizon, pred_config_dict, predictions_dir, iteration = args
    
    try:
        # Recreate pred_config from dict
        from utils.config_loader import load_prediction_models_config
        pred_config = load_prediction_models_config()
        
        # Load training data
        df = load_dataset(file_info['path'])
        train_series = df.iloc[:, 0].dropna()
        
        if len(train_series) < 10:
            return None
        
        # Start monitoring
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Make predictions
        if model_name == 'xgboost':
            # Load XGBoost model
            models_dir = pred_config_dict.get('models_dir', 'trained_prediction_models')
            pkl_path = os.path.join(models_dir, f"{model_name}_iter{iteration}.pkl")
            with open(pkl_path, 'rb') as f:
                trained_model = pickle.load(f)
            predictions = predict_with_xgboost(trained_model, train_series, horizon)
        else:
            predictions = predict_with_statistical_model(
                model_name, train_series, horizon, pred_config
            )
        
        # Stop monitoring
        metrics = monitor.stop()
        
        # Generate output filename
        if file_info['source_type'] == 'original':
            base_name = f"{file_info['dataset']}_original_{model_name}"
        else:
            base_name = (f"{file_info['dataset']}_{file_info['technique']}_"
                        f"{file_info['rate_percent']}p_{file_info['reconstruction_iteration']}_"
                        f"{file_info['reconstruction_model']}_{model_name}")
        
        output_filename = f"{base_name}.csv"
        output_path = os.path.join(predictions_dir, output_filename)
        
        # Save predictions
        save_predictions(predictions, output_path, train_series, iteration)
        
        # Return metrics
        return {
            'dataset_name': file_info['dataset'],
            'source_type': file_info['source_type'],
            'technique': file_info['technique'],
            'rate_percent': file_info['rate_percent'],
            'reconstruction_iteration': file_info['reconstruction_iteration'],
            'reconstruction_model': file_info['reconstruction_model'],
            'prediction_model': model_name,
            'prediction_iteration': iteration,
            'time_seconds': metrics['time_seconds'],
            'cpu_cores_used': metrics['cpu_cores_used'],
            'cpu_cores_total': metrics['cpu_cores_total'],
            'memory_mb': metrics['memory_mb'],
            'memory_total_mb': metrics['memory_total_mb'],
            'gpu_percent': metrics.get('gpu_percent'),
            'gpu_memory_mb': metrics.get('gpu_memory_mb'),
            'gpu_memory_total_mb': metrics.get('gpu_memory_total_mb'),
        }
        
    except Exception as e:
        # Return error info
        return {'error': str(e), 'dataset': file_info.get('dataset', 'unknown')}


def generate_metrics_filename():
    """Generate output filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"prediction_metrics_{timestamp}.csv"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Predict datasets using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict using all available trained models
  python 8_predict_datasets.py
  
  # Predict using specific models only
  python 8_predict_datasets.py --models lstm xgboost holt_winters
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to main configuration file'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        default=None,
        help='Specific models to use for prediction'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='trained_prediction_models',
        help='Directory with trained models'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    try:
        config = load_config(args.config)
        pred_config = load_prediction_models_config()
        print(f"✓ Loaded configuration from: {args.config}")
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        return
    
    # Get directories
    train_dir = config.get_splitted_train_dir()
    test_dir = config.get_splitted_test_dir()
    fixed_dir = config.get_fixed_dir()
    models_dir = args.models_dir
    output_dir = config.get_prediction_results_dir()
    predictions_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "performance_metrics")
    
    # Create output directories
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Get available trained models
    available_models = get_available_trained_models(models_dir)
    
    # Get statistical models (trained per-file)
    statistical_models = pred_config.get_per_file_training_models()
    
    # Determine which models to use
    all_models = list(available_models.keys()) + statistical_models
    
    if args.models:
        # Command line override
        selected_models = [m for m in args.models if m in all_models]
        if not selected_models:
            print(f"❌ No valid models specified. Available: {all_models}")
            return
    else:
        # Use config.get_prediction_models() which reads prediction_models.selected
        config_selected = config.get_prediction_models()
        if config_selected:
            selected_models = [m for m in config_selected if m in all_models]
            if not selected_models:
                print(f"⚠️ Config selected models not available, using all: {all_models}")
                selected_models = all_models
        else:
            # Use all available models
            selected_models = all_models
    
    # Get prediction horizon from test data
    test_files = list(Path(test_dir).glob("*.csv"))
    if test_files:
        test_df = load_dataset(str(test_files[0]))
        horizon = len(test_df)
    else:
        horizon = 30
    
    print("="*70)
    print("PREDICT DATASETS")
    print("="*70)
    print(f"Trained models directory: {models_dir}")
    print(f"Available trained models: {list(available_models.keys())}")
    print(f"Statistical models (per-file): {statistical_models}")
    print(f"Selected models: {selected_models}")
    print(f"Prediction horizon: {horizon}")
    print(f"Output directory: {predictions_dir}")
    print("="*70)
    
    # Collect all files to predict
    files_to_predict = []
    
    # Original training files
    if config.get_predict_on_original_train():
        for file in Path(train_dir).glob("*.csv"):
            files_to_predict.append({
                'path': str(file),
                'source_type': 'original',
                'dataset': file.stem,
                'technique': None,
                'rate_percent': None,
                'reconstruction_iteration': None,
                'reconstruction_model': None
            })
    
    # Reconstructed files
    if config.get_predict_on_reconstructed() and os.path.exists(fixed_dir):
        for file in Path(fixed_dir).glob("*.csv"):
            try:
                metadata = parse_reconstructed_filename(file.name)
                files_to_predict.append({
                    'path': str(file),
                    'source_type': 'reconstructed',
                    'dataset': metadata['dataset'],
                    'technique': metadata['technique'],
                    'rate_percent': metadata['rate_percent'],
                    'reconstruction_iteration': metadata['iteration'],
                    'reconstruction_model': metadata['reconstruction_model']
                })
            except ValueError:
                pass
    
    print(f"\n📊 Files to predict: {len(files_to_predict)}")
    
    # Storage for prediction metrics
    prediction_metrics = []
    
    # CPU models that can be parallelized
    cpu_models = set(statistical_models) | {'xgboost'}
    num_workers = max(1, (os.cpu_count() or 1) - 1)
    
    # Process each model
    for model_name in selected_models:
        is_statistical = model_name in statistical_models
        is_trained = model_name in available_models
        is_cpu_model = model_name in cpu_models
        
        if is_statistical:
            # Statistical models - train per-file
            iterations = [1]  # Only one iteration for deterministic models
        elif is_trained:
            iterations = available_models[model_name]
        else:
            print(f"⚠️ Model {model_name} not found, skipping")
            continue
        
        for iteration in iterations:
            desc = f"{model_name.upper()}" + (f" iter{iteration}" if len(iterations) > 1 else "")
            
            # Use parallel processing for CPU models
            if is_cpu_model:
                # Prepare arguments for parallel processing
                pred_config_dict = {'models_dir': models_dir}
                args_list = [
                    (file_info, model_name, horizon, pred_config_dict, predictions_dir, iteration)
                    for file_info in files_to_predict
                ]
                
                total_files = len(args_list)
                print(f"\n🚀 {desc}: {total_files} files, {num_workers} workers")
                
                # Process in parallel with progress tracking
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    futures = [executor.submit(process_single_file_statistical, args) 
                              for args in args_list]
                    
                    completed = 0
                    with tqdm(total=total_files, desc=f"{desc}", unit="file", leave=True) as pbar:
                        while completed < total_files:
                            # Count running and completed
                            running = sum(1 for f in futures if f.running())
                            done_now = sum(1 for f in futures if f.done())
                            
                            # Update progress bar
                            new_completed = done_now - completed
                            if new_completed > 0:
                                pbar.update(new_completed)
                                completed = done_now
                            
                            # Update description with active workers
                            pbar.set_postfix({'active': running}, refresh=True)
                            
                            if completed < total_files:
                                import time
                                time.sleep(0.1)
                    
                    # Collect results
                    for future in futures:
                        result = future.result()
                        if result is not None:
                            if 'error' in result:
                                print(f"\n   ❌ Error: {result['error']}")
                                raise RuntimeError(f"Prediction failed: {result['error']}")
                            else:
                                prediction_metrics.append(result)
            else:
                # GPU models - process sequentially
                trained_model = None
                try:
                    trained_model = load_trained_model(model_name, iteration, models_dir)
                except FileNotFoundError as e:
                    print(f"❌ {e}")
                    continue
                
                for file_info in tqdm(files_to_predict, desc=desc, unit="file", leave=True):
                    try:
                        # Load training data
                        df = load_dataset(file_info['path'])
                        train_series = df.iloc[:, 0].dropna()
                        
                        if len(train_series) < 10:
                            continue
                        
                        # Start monitoring
                        monitor = PerformanceMonitor()
                        monitor.start()
                        
                        # Make predictions with Darts model
                        predictions = predict_with_darts_model(
                            trained_model, train_series, horizon
                        )
                        
                        # Stop monitoring
                        metrics = monitor.stop()
                        
                        # Generate output filename
                        if file_info['source_type'] == 'original':
                            base_name = f"{file_info['dataset']}_original_{model_name}"
                        else:
                            base_name = (f"{file_info['dataset']}_{file_info['technique']}_"
                                        f"{file_info['rate_percent']}p_{file_info['reconstruction_iteration']}_"
                                        f"{file_info['reconstruction_model']}_{model_name}")
                        
                        if len(iterations) > 1:
                            output_filename = f"{base_name}_iter{iteration}.csv"
                        else:
                            output_filename = f"{base_name}.csv"
                        
                        output_path = os.path.join(predictions_dir, output_filename)
                        
                        # Save predictions
                        save_predictions(predictions, output_path, train_series, iteration)
                        
                        # Store metrics
                        prediction_metrics.append({
                            'dataset_name': file_info['dataset'],
                            'source_type': file_info['source_type'],
                            'technique': file_info['technique'],
                            'rate_percent': file_info['rate_percent'],
                            'reconstruction_iteration': file_info['reconstruction_iteration'],
                            'reconstruction_model': file_info['reconstruction_model'],
                            'prediction_model': model_name,
                            'prediction_iteration': iteration,
                            'time_seconds': metrics['time_seconds'],
                            'cpu_cores_used': metrics['cpu_cores_used'],
                            'cpu_cores_total': metrics['cpu_cores_total'],
                            'memory_mb': metrics['memory_mb'],
                            'memory_total_mb': metrics['memory_total_mb'],
                            'gpu_percent': metrics.get('gpu_percent'),
                            'gpu_memory_mb': metrics.get('gpu_memory_mb'),
                            'gpu_memory_total_mb': metrics.get('gpu_memory_total_mb'),
                        })
                        
                    except Exception as e:
                        print(f"\n   ❌ Error predicting {file_info['dataset']}: {e}")
                        raise  # Stop on error
    
    # Save prediction metrics
    metrics_filename = generate_metrics_filename()
    metrics_path = os.path.join(metrics_dir, metrics_filename)
    
    df_metrics = pd.DataFrame(prediction_metrics)
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"✓ Predictions saved to: {predictions_dir}/")
    print(f"✓ Metrics saved to: {metrics_path}")
    print(f"\n📊 Summary:")
    print(f"   Total predictions: {len(prediction_metrics)}")
    if len(df_metrics) > 0 and 'time_seconds' in df_metrics.columns:
        print(f"   Total time: {df_metrics['time_seconds'].sum():.2f}s")
    print("="*70)


if __name__ == "__main__":
    main()

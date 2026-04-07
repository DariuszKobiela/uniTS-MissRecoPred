#!/usr/bin/env python3
"""
Train Prediction Models Script
Trains all prediction models that require training (deep learning + XGBoost).

This script:
1. Trains global models (LSTM, GRU, TCN, N-BEATS, DeepAR, Vanilla Transformer, TFT, XGBoost)
2. Trains N times for non-deterministic models (configurable)
3. Saves trained models to trained_models/ folder
4. Saves training metrics (time, CPU, GPU) to training_metrics_*.csv

Statistical models (SARIMAX, Holt-Winters, Prophet) are NOT trained here
because they require per-file training, which is done during prediction.

Uses config/config.yaml and config/prediction_models_config.yaml for configuration.
"""

import os
import sys
import warnings

# Suppress NVML warnings from PyTorch before any torch imports
warnings.filterwarnings('ignore', message='.*NVML.*')
warnings.filterwarnings('ignore', message='.*Can\'t initialize.*')

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import gc
import pickle
from concurrent.futures import ThreadPoolExecutor
import torch

# Optimize for GPUs with Tensor Cores (RTX, A100, etc.)
torch.set_float32_matmul_precision('medium')

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import config loader
from utils.config_loader import load_config, load_prediction_models_config
from utils.performance_metrics import PerformanceMonitor, format_metrics
from utils.logger import setup_logging, EpochLogger

# Setup automatic logging to file
setup_logging("7_train_prediction_models")


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


def _load_series_from_file(file_path: str) -> Optional[pd.Series]:
    """
    Load a single CSV file and extract its first column as a Series.
    Returns None if loading fails or series is too short.
    Used by parallel CSV loader.
    """
    try:
        df = load_dataset(file_path)
        series = df.iloc[:, 0].dropna()
        if len(series) > 10:
            return series
    except Exception:
        pass
    return None


def load_series_parallel(file_paths: List[str], max_workers: int = 8) -> List[pd.Series]:
    """
    Load multiple CSV files in parallel using threads (I/O-bound).
    
    Args:
        file_paths: List of CSV file paths
        max_workers: Number of parallel threads
        
    Returns:
        List of pandas Series (filtered: non-None, len > 10)
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(tqdm(
            executor.map(_load_series_from_file, file_paths),
            total=len(file_paths),
            desc="   Loading CSVs"
        ))
    
    results = [s for s in futures if s is not None]
    return results


def series_to_darts(series: pd.Series):
    """Convert pandas Series to Darts TimeSeries (using float32 to save ~50% RAM)."""
    from darts import TimeSeries
    
    date_index = pd.date_range(start='2000-01-01', periods=len(series), freq='h')
    ts = TimeSeries.from_times_and_values(
        times=date_index, values=series.values.astype(np.float32), freq='h'
    )
    return ts


def prepare_darts_training_data(all_series: List[pd.Series], val_split: float):
    """
    Pre-convert all pandas Series to Darts TimeSeries with train/val split.
    
    Done ONCE in main() to avoid repeated conversion per model,
    which saves ~5 GB RAM and significant CPU time.
    
    Args:
        all_series: List of pandas Series (raw time series)
        val_split: Fraction of each series to use for validation (0.0 to 1.0)
        
    Returns:
        Tuple of (train_series_list, val_series_list) as Darts TimeSeries
    """
    train_series_list = []
    val_series_list = []
    
    for series in tqdm(all_series, desc="   Converting to Darts"):
        ts = series_to_darts(series)
        
        # Split into train/val
        if val_split > 0 and len(ts) > 10:
            split_point = int(len(ts) * (1 - val_split))
            train_ts = ts[:split_point]
            val_ts = ts[split_point:]
            
            if len(train_ts) > 5 and len(val_ts) > 0:
                train_series_list.append(train_ts)
                val_series_list.append(val_ts)
            else:
                train_series_list.append(ts)
        else:
            train_series_list.append(ts)
    
    return train_series_list, val_series_list


def train_global_model_darts(model_name: str,
                              train_series_list: List,
                              val_series_list: List,
                              pred_config,
                              seed: int = None):
    """
    Train a global Darts model on all time series.
    
    Args:
        model_name: Name of the model (lstm, gru, tcn, nbeats, deepar, vanilla_transformer, temporal_fusion_transformer)
        train_series_list: Pre-converted Darts TimeSeries for training
        val_series_list: Pre-converted Darts TimeSeries for validation
        pred_config: PredictionModelsConfig object
        seed: Random seed for this training iteration
        
    Returns:
        Tuple of (trained_model, training_info) where training_info is a dict with:
            - epochs_trained: actual number of epochs completed
            - best_val_loss: lowest validation loss (or None)
            - final_train_loss: training loss at last epoch (or None)
    """
    from darts import TimeSeries
    from darts.models import RNNModel, TCNModel, NBEATSModel, TFTModel, TransformerModel
    from darts.utils.likelihood_models import GaussianLikelihood
    from pytorch_lightning.callbacks import EarlyStopping
    
    # Get global training parameters
    max_epochs = pred_config.get_max_epochs()
    batch_size = pred_config.get_model_batch_size(model_name)
    dataloader_kwargs = pred_config.get_dataloader_kwargs()
    
    # Get early stopping parameters
    es_enabled = pred_config.get_early_stopping_enabled()
    es_patience = pred_config.get_early_stopping_patience()
    es_min_delta = pred_config.get_early_stopping_min_delta()
    es_verbose = pred_config.get_early_stopping_verbose()
    
    # Get model-specific parameters
    model_params = pred_config.get_model_params(model_name)
    
    # Set up callbacks (keep reference to epoch_logger for post-training stats)
    epoch_logger = EpochLogger()
    callbacks = [epoch_logger]
    if es_enabled:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            min_delta=es_min_delta,
            mode="min",
            verbose=es_verbose
        )
        callbacks.append(early_stopping)
    
    # Create model based on type
    input_chunk_length = model_params.get('input_chunk_length', 24)
    
    # Adjust input_chunk_length if series are too short
    min_series_len = min(len(ts) for ts in train_series_list)
    if min_series_len < input_chunk_length:
        input_chunk_length = max(10, min_series_len // 2)
    
    pl_trainer_kwargs = {
        "callbacks": callbacks,
        "accelerator": "auto",
        "enable_progress_bar": True,   # Show in console (filtered from log file)
        "enable_model_summary": False,
    }
    
    if model_name in ['lstm', 'gru']:
        model = RNNModel(
            model=model_name.upper(),
            input_chunk_length=input_chunk_length,
            training_length=max(model_params.get('training_length', 30), input_chunk_length),
            hidden_dim=model_params.get('hidden_dim', 32),
            n_rnn_layers=model_params.get('n_layers', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=True
        )
    elif model_name == 'deepar':
        model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            training_length=max(model_params.get('training_length', 30), input_chunk_length),
            hidden_dim=model_params.get('hidden_dim', 40),
            n_rnn_layers=model_params.get('n_layers', 2),
            dropout=model_params.get('dropout', 0.1),
            likelihood=GaussianLikelihood(),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=True
        )
    elif model_name == 'tcn':
        output_chunk_length = min(model_params.get('output_chunk_length', 6), min_series_len // 4)
        model = TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=max(1, output_chunk_length),
            kernel_size=model_params.get('kernel_size', 3),
            num_filters=model_params.get('num_filters', 64),
            dilation_base=model_params.get('dilation_base', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=True
        )
    elif model_name == 'nbeats':
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        model = NBEATSModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=max(1, output_chunk_length),
            generic_architecture=model_params.get('generic_architecture', True),
            num_stacks=model_params.get('num_stacks', 30),
            num_blocks=model_params.get('num_blocks', 1),
            num_layers=model_params.get('num_layers', 4),
            layer_widths=model_params.get('layer_widths', 256),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=True
        )
    elif model_name == 'vanilla_transformer':
        # Vanilla Transformer (encoder-decoder self-attention)
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        model = TransformerModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=max(1, output_chunk_length),
            d_model=model_params.get('d_model', 64),
            nhead=model_params.get('nhead', 4),
            num_encoder_layers=model_params.get('num_encoder_layers', 2),
            num_decoder_layers=model_params.get('num_decoder_layers', 2),
            dim_feedforward=model_params.get('dim_feedforward', 128),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=True
        )
    elif model_name == 'temporal_fusion_transformer':
        # Temporal Fusion Transformer (specialized for forecasting)
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        model = TFTModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=max(1, output_chunk_length),
            hidden_size=model_params.get('hidden_size', 64),
            lstm_layers=model_params.get('lstm_layers', 1),
            num_attention_heads=model_params.get('num_attention_heads', 4),
            dropout=model_params.get('dropout', 0.1),
            add_relative_index=model_params.get('add_relative_index', True),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=True
        )
    else:
        raise ValueError(f"Unknown global training model: {model_name}")
    
    # Train the model on all series
    dl_info = f", dataloader: {dataloader_kwargs}" if dataloader_kwargs else ""
    print(f"   Starting training on {len(train_series_list)} series "
          f"(batch_size={batch_size}{dl_info})...", flush=True)
    
    fit_kwargs = {"verbose": True}
    if dataloader_kwargs:
        fit_kwargs["dataloader_kwargs"] = dataloader_kwargs
    
    if val_series_list:
        model.fit(train_series_list, val_series=val_series_list, **fit_kwargs)
    else:
        model.fit(train_series_list, **fit_kwargs)
    
    # Collect training info from EpochLogger
    training_info = {
        'epochs_trained': epoch_logger.epochs_trained,
        'best_val_loss': epoch_logger.best_val_loss,
        'final_train_loss': epoch_logger.final_train_loss,
    }
    
    return model, training_info


def _create_lag_features_vectorized(values: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lag features from a single time series using numpy stride tricks.
    
    ~10-50x faster than the equivalent Python loop for large arrays.
    
    Args:
        values: 1D numpy array of time series values
        lag: Number of lag features
        
    Returns:
        Tuple of (X, y) where X has shape (n_samples, lag) and y has shape (n_samples,)
    """
    n = len(values)
    if n <= lag:
        return np.empty((0, lag), dtype=values.dtype), np.empty(0, dtype=values.dtype)
    
    # Use stride tricks for zero-copy sliding window view
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(values, lag + 1)  # shape: (n - lag, lag + 1)
    X = windows[:, :lag]   # first `lag` columns = features
    y = windows[:, lag]    # last column = target
    
    return X, y


def train_xgboost_model(all_values: List[np.ndarray], pred_config, seed: int = None):
    """
    Train a global XGBoost model using lag features.
    
    Args:
        all_values: List of numpy arrays (time series values, float32)
        pred_config: PredictionModelsConfig object
        seed: Random seed
        
    Returns:
        Tuple of (trained model, lag count)
    """
    import xgboost as xgb
    
    model_params = pred_config.get_model_params('xgboost')
    lag = model_params.get('lags', 10)
    
    # CPU OPTIMIZATION: Vectorized lag feature creation with numpy stride tricks
    # ~10-50x faster than nested Python loops for large datasets
    X_parts = []
    y_parts = []
    
    for values in all_values:
        X_part, y_part = _create_lag_features_vectorized(values, lag)
        if len(X_part) > 0:
            X_parts.append(X_part)
            y_parts.append(y_part)
    
    if len(X_parts) == 0:
        raise ValueError("Not enough data to create lag features")
    
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    
    # Free intermediate lists
    del X_parts, y_parts
    
    # Train XGBoost (n_jobs=-1 uses all CPU cores for parallel tree building)
    model = xgb.XGBRegressor(
        n_estimators=model_params.get('n_estimators', 100),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.1),
        n_jobs=model_params.get('n_jobs', -1),
        random_state=seed,
        verbosity=0
    )
    model.fit(X, y)
    
    return model, lag


def get_model_path(model_name: str, iteration: int, output_dir: str, model_type: str = 'darts') -> str:
    """Get the path where a model would be saved."""
    if model_type == 'darts':
        return os.path.join(output_dir, f"{model_name}_iter{iteration}.pt")
    else:
        return os.path.join(output_dir, f"{model_name}_iter{iteration}.pkl")


def model_exists(model_name: str, iteration: int, output_dir: str) -> bool:
    """Check if a trained model already exists."""
    # Check for Darts model (.pt)
    darts_path = os.path.join(output_dir, f"{model_name}_iter{iteration}.pt")
    if os.path.exists(darts_path):
        return True
    
    # Check for pickle model (.pkl)
    pkl_path = os.path.join(output_dir, f"{model_name}_iter{iteration}.pkl")
    if os.path.exists(pkl_path):
        return True
    
    return False


def save_model(model, model_name: str, iteration: int, output_dir: str, model_type: str = 'darts'):
    """Save trained model to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'darts':
        # Darts models have their own save method
        model_path = os.path.join(output_dir, f"{model_name}_iter{iteration}.pt")
        model.save(model_path)
    else:
        # XGBoost and other models - use pickle
        model_path = os.path.join(output_dir, f"{model_name}_iter{iteration}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return model_path


def generate_training_metrics_filename(model_name: str = None, iteration: int = None):
    """
    Generate output filename with timestamp, model name and iteration.
    
    Each trained model+iteration gets its own metrics file, saved immediately
    after training (so metrics are not lost if the process crashes later).
    
    Format: training_metrics_YYYYMMDD_HHMMSS_<model>_iter<N>.csv
    
    Args:
        model_name: Name of the model (e.g. 'lstm', 'nbeats')
        iteration: Iteration number
        
    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name is not None and iteration is not None:
        return f"training_metrics_{timestamp}_{model_name}_iter{iteration}.csv"
    return f"training_metrics_{timestamp}.csv"


def save_training_metrics(metrics_dict: dict, metrics_dir: str, model_name: str, iteration: int):
    """
    Save training metrics for a single model+iteration to its own CSV file.
    
    Args:
        metrics_dict: Dictionary with metric values
        metrics_dir: Directory to save the file in
        model_name: Name of the trained model
        iteration: Iteration number
        
    Returns:
        Path to the saved metrics file
    """
    os.makedirs(metrics_dir, exist_ok=True)
    filename = generate_training_metrics_filename(model_name, iteration)
    filepath = os.path.join(metrics_dir, filename)
    
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filepath, index=False)
    
    return filepath


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models using default config (skips existing)
  python 7_train_prediction_models.py
  
  # Train specific models only
  python 7_train_prediction_models.py --models lstm gru xgboost
  
  # Force retrain all models (overwrite existing)
  python 7_train_prediction_models.py --force
  
  # Train with specific number of iterations
  python 7_train_prediction_models.py --iterations 5
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
        help='Specific models to train (default: all trainable models)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Override number of training iterations'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if models already exist (overwrite)'
    )
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        pred_config = load_prediction_models_config()
        print(f"✓ Loaded configuration from: {args.config}")
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        return

    run_train_prediction_models(
        config,
        pred_config,
        models=args.models,
        iterations=args.iterations,
        force=args.force,
    )


def run_train_prediction_models(config, pred_config, models=None, iterations=None, force=False) -> bool:
    """Step 7: train prediction models (Darts + XGBoost)."""
    train_dir = config.get_splitted_train_dir()
    fixed_dir = config.get_fixed_dir()
    output_dir = "trained_prediction_models"
    metrics_dir = "prediction_experiment_results"

    if not os.path.exists(train_dir):
        print(f"❌ Training data directory not found: {train_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    global_models = pred_config.get_global_training_models()
    ml_models = pred_config.get_ml_models()
    trainable_models = global_models + ml_models

    if models:
        trainable_models = [m for m in models if m in trainable_models]
        if not trainable_models:
            print(f"❌ No valid trainable models specified. Available: {global_models + ml_models}")
            return False

    n_iterations = iterations if iterations is not None else pred_config.get_training_iterations()
    base_seed = pred_config.get_seed()

    deterministic_models = set(pred_config.get_deterministic_models())

    overwrite = force or config.get_overwrite_prediction()

    print("="*70)
    print("TRAIN PREDICTION MODELS")
    print("="*70)
    print(f"Trainable models: {trainable_models}")
    print(f"Training iterations: {n_iterations}")
    print(f"Base seed: {base_seed}")
    print(f"Output directory: {output_dir}")
    print(f"Overwrite existing: {overwrite} (config: {config.get_overwrite_prediction()}, --force: {force})")
    print("="*70)

    print("\n📋 Collecting training data...")

    all_series = []
    csv_files = []

    if config.get_predict_on_original_train():
        train_files = sorted(Path(train_dir).glob("*.csv"))
        csv_files.extend([str(f) for f in train_files])
        print(f"   Found {len(train_files)} files in {train_dir}")

    if config.get_predict_on_reconstructed() and os.path.exists(fixed_dir):
        fixed_files = sorted(Path(fixed_dir).glob("*.csv"))
        csv_files.extend([str(f) for f in fixed_files])
        print(f"   Found {len(fixed_files)} files in {fixed_dir}")

    if csv_files:
        n_loader_threads = min(16, max(1, os.cpu_count() or 1))
        print(f"   Loading {len(csv_files)} CSV files in parallel ({n_loader_threads} threads)...")
        all_series = load_series_parallel(csv_files, max_workers=n_loader_threads)

    print(f"\n📊 Total training series: {len(all_series)}")

    if len(all_series) == 0:
        print("❌ No training data found")
        return False

    print("\n🔧 Converting series to float32 to reduce memory usage...")
    for i in range(len(all_series)):
        all_series[i] = all_series[i].astype(np.float32)
    gc.collect()
    print(f"   ✓ Converted {len(all_series)} series to float32")

    xgboost_values = None
    if 'xgboost' in trainable_models:
        xgboost_values = [s.values.copy() for s in all_series]

    val_split = pred_config.get_validation_split()

    darts_models_to_train = [m for m in trainable_models if m in global_models]

    darts_train = []
    darts_val = []

    if darts_models_to_train:
        print("\n🔧 Pre-converting to Darts TimeSeries (one-time conversion)...")
        darts_train, darts_val = prepare_darts_training_data(all_series, val_split)
        print(f"   ✓ Train series: {len(darts_train)}, Val series: {len(darts_val)}")

    del all_series
    gc.collect()
    print("   ✓ Freed original pandas Series from memory")

    training_metrics = []
    saved_metrics_files = []

    for model_name in trainable_models:
        is_deterministic = model_name in deterministic_models
        model_iterations = 1 if is_deterministic else n_iterations

        print(f"\n{'='*70}")
        print(f"Training: {model_name.upper()}")
        batch_size_info = pred_config.get_model_batch_size(model_name)
        print(f"Iterations: {model_iterations}, Batch size: {batch_size_info}")
        print(f"{'='*70}")

        for iteration in range(1, model_iterations + 1):
            seed = base_seed + iteration

            if model_exists(model_name, iteration, output_dir) and not overwrite:
                existing_path = get_model_path(model_name, iteration, output_dir,
                                               'darts' if model_name in global_models else 'pickle')
                print(f"\n⏭️  Skipping {model_name} iter{iteration} (already exists: {existing_path})")
                print(f"   Set overwrite.prediction=true in config or use --force to retrain")
                continue

            print(f"\n🔧 Training {model_name} (iteration {iteration}/{model_iterations}, seed={seed})...")

            monitor = PerformanceMonitor()
            monitor.start()

            try:
                training_info = {}

                if model_name in global_models:
                    model, training_info = train_global_model_darts(
                        model_name, darts_train, darts_val, pred_config, seed
                    )
                    model_path = save_model(model, model_name, iteration, output_dir, 'darts')

                elif model_name == 'xgboost':
                    model, lag = train_xgboost_model(xgboost_values, pred_config, seed)
                    model_data = {'model': model, 'lag': lag}
                    model_path = os.path.join(output_dir, f"{model_name}_iter{iteration}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data, f)

                metrics = monitor.stop()

                print(f"   ✓ Saved: {model_path}")
                print(f"   ⏱️ Time: {metrics['time_seconds']:.2f}s")
                if training_info.get('epochs_trained'):
                    epochs = training_info['epochs_trained']
                    val_loss_str = (f", best_val_loss={training_info['best_val_loss']:.6f}"
                                    if training_info.get('best_val_loss') is not None else "")
                    print(f"   📈 Epochs: {epochs}{val_loss_str}")

                metrics_dict = {
                    'model': model_name,
                    'iteration': iteration,
                    'seed': seed,
                    'epochs_trained': training_info.get('epochs_trained'),
                    'best_val_loss': training_info.get('best_val_loss'),
                    'final_train_loss': training_info.get('final_train_loss'),
                    'time_seconds': metrics['time_seconds'],
                    'cpu_cores_used': metrics['cpu_cores_used'],
                    'cpu_cores_total': metrics['cpu_cores_total'],
                    'memory_mb': metrics['memory_mb'],
                    'memory_total_mb': metrics['memory_total_mb'],
                    'gpu_percent': metrics.get('gpu_percent'),
                    'gpu_memory_mb': metrics.get('gpu_memory_mb'),
                    'gpu_memory_total_mb': metrics.get('gpu_memory_total_mb'),
                    'model_path': model_path
                }

                metrics_file = save_training_metrics(
                    metrics_dict, metrics_dir, model_name, iteration
                )
                saved_metrics_files.append(metrics_file)
                print(f"   📊 Metrics: {metrics_file}")

                training_metrics.append(metrics_dict)

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                monitor.stop()
                print(f"   ❌ Failed to train {model_name}: {e}")

                error_metrics = {
                    'model': model_name,
                    'iteration': iteration,
                    'seed': seed,
                    'time_seconds': None,
                    'error': str(e)
                }
                metrics_file = save_training_metrics(
                    error_metrics, metrics_dir, model_name, iteration
                )
                saved_metrics_files.append(metrics_file)

                training_metrics.append(error_metrics)
                gc.collect()

    if xgboost_values is not None:
        del xgboost_values
        gc.collect()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Models saved to: {output_dir}/")
    print(f"✓ Metrics files saved to: {metrics_dir}/")
    if saved_metrics_files:
        for mf in saved_metrics_files:
            print(f"   - {mf}")

    print(f"\n📊 Summary:")
    if training_metrics:
        df_metrics = pd.DataFrame(training_metrics)
        successful = df_metrics[df_metrics['time_seconds'].notna()]
        print(f"   Total trained: {len(successful)}/{len(df_metrics)}")
        if len(successful) > 0:
            print(f"   Total time: {successful['time_seconds'].sum():.2f}s")
            print(f"   Avg time per model: {successful['time_seconds'].mean():.2f}s")
    else:
        print(f"   No models were trained (all models may have been skipped or already exist)")
    print("="*70)
    return True


if __name__ == "__main__":
    main()

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
import pickle
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


def series_to_darts(series: pd.Series):
    """Convert pandas Series to Darts TimeSeries."""
    from darts import TimeSeries
    
    date_index = pd.date_range(start='2000-01-01', periods=len(series), freq='h')
    ts = TimeSeries.from_times_and_values(times=date_index, values=series.values, freq='h')
    return ts


def train_global_model_darts(model_name: str,
                              all_series: List[pd.Series],
                              pred_config,
                              seed: int = None):
    """
    Train a global Darts model on all time series.
    
    Args:
        model_name: Name of the model (lstm, gru, tcn, nbeats, deepar, vanilla_transformer, temporal_fusion_transformer)
        all_series: List of all training time series
        pred_config: PredictionModelsConfig object
        seed: Random seed for this training iteration
        
    Returns:
        Trained Darts model
    """
    from darts import TimeSeries
    from darts.models import RNNModel, TCNModel, NBEATSModel, TFTModel, TransformerModel
    from darts.utils.likelihood_models import GaussianLikelihood
    from pytorch_lightning.callbacks import EarlyStopping
    
    # Get global training parameters
    val_split = pred_config.get_validation_split()
    max_epochs = pred_config.get_max_epochs()
    batch_size = pred_config.get_batch_size()
    
    # Get early stopping parameters
    es_enabled = pred_config.get_early_stopping_enabled()
    es_patience = pred_config.get_early_stopping_patience()
    es_min_delta = pred_config.get_early_stopping_min_delta()
    es_verbose = pred_config.get_early_stopping_verbose()
    
    # Get model-specific parameters
    model_params = pred_config.get_model_params(model_name)
    
    # Convert all series to Darts TimeSeries
    train_series_list = []
    val_series_list = []
    
    for series in all_series:
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
    
    # Set up callbacks
    callbacks = [EpochLogger()]  # Always log epoch numbers
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
    print(f"   Starting training on {len(train_series_list)} series...", flush=True)
    if val_series_list:
        model.fit(train_series_list, val_series=val_series_list, verbose=True)
    else:
        model.fit(train_series_list, verbose=True)
    
    return model


def train_xgboost_model(all_series: List[pd.Series], pred_config, seed: int = None):
    """
    Train a global XGBoost model using lag features.
    
    Args:
        all_series: List of all training time series
        pred_config: PredictionModelsConfig object
        seed: Random seed
        
    Returns:
        Tuple of (trained model, lag count)
    """
    import xgboost as xgb
    
    model_params = pred_config.get_model_params('xgboost')
    lag = model_params.get('lags', 10)
    
    # Create lag features from all series
    X_all = []
    y_all = []
    
    for series in all_series:
        values = series.values
        if len(values) <= lag:
            continue
        
        for i in range(lag, len(values)):
            X_all.append(values[i-lag:i])
            y_all.append(values[i])
    
    if len(X_all) == 0:
        raise ValueError("Not enough data to create lag features")
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=model_params.get('n_estimators', 100),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.1),
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


def generate_training_metrics_filename():
    """Generate output filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"training_metrics_{timestamp}.csv"


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
    fixed_dir = config.get_fixed_dir()
    output_dir = "trained_prediction_models"
    metrics_dir = "prediction_experiment_results"
    
    # Check directories exist
    if not os.path.exists(train_dir):
        print(f"❌ Training data directory not found: {train_dir}")
        return
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Get models to train
    global_models = pred_config.get_global_training_models()
    ml_models = pred_config.get_ml_models()
    trainable_models = global_models + ml_models
    
    if args.models:
        trainable_models = [m for m in args.models if m in trainable_models]
        if not trainable_models:
            print(f"❌ No valid trainable models specified. Available: {global_models + ml_models}")
            return
    
    # Get training iterations
    n_iterations = args.iterations if args.iterations else pred_config.get_training_iterations()
    base_seed = pred_config.get_seed()
    
    # Determine which models are deterministic
    deterministic_models = set(pred_config.get_deterministic_models())
    
    # Determine overwrite setting (--force arg takes priority over config)
    overwrite = args.force or config.get_overwrite_prediction()
    
    print("="*70)
    print("TRAIN PREDICTION MODELS")
    print("="*70)
    print(f"Trainable models: {trainable_models}")
    print(f"Training iterations: {n_iterations}")
    print(f"Base seed: {base_seed}")
    print(f"Output directory: {output_dir}")
    print(f"Overwrite existing: {overwrite} (config: {config.get_overwrite_prediction()}, --force: {args.force})")
    print("="*70)
    
    # Collect all training data
    print("\n📋 Collecting training data...")
    
    all_series = []
    
    # Load original training data
    print(f"   Loading from {train_dir}...")
    if config.get_predict_on_original_train():
        for file in Path(train_dir).glob("*.csv"):
            try:
                df = load_dataset(str(file))
                series = df.iloc[:, 0].dropna()
                if len(series) > 10:
                    all_series.append(series)
            except Exception as e:
                print(f"   ⚠️ Error loading {file.name}: {e}")
    
    # Load reconstructed data
    print(f"   Loading from {fixed_dir}...")
    if config.get_predict_on_reconstructed() and os.path.exists(fixed_dir):
        for file in Path(fixed_dir).glob("*.csv"):
            try:
                df = load_dataset(str(file))
                series = df.iloc[:, 0].dropna()
                if len(series) > 10:
                    all_series.append(series)
            except Exception as e:
                print(f"   ⚠️ Error loading {file.name}: {e}")
    
    print(f"\n📊 Total training series: {len(all_series)}")
    
    if len(all_series) == 0:
        print("❌ No training data found")
        return
    
    # Training metrics storage
    training_metrics = []
    
    # Train each model
    for model_name in trainable_models:
        is_deterministic = model_name in deterministic_models
        model_iterations = 1 if is_deterministic else n_iterations
        
        print(f"\n{'='*70}")
        print(f"Training: {model_name.upper()}")
        print(f"Iterations: {model_iterations}")
        print(f"{'='*70}")
        
        for iteration in range(1, model_iterations + 1):
            seed = base_seed + iteration
            
            # Check if model already exists (skip unless overwrite=True)
            if model_exists(model_name, iteration, output_dir) and not overwrite:
                existing_path = get_model_path(model_name, iteration, output_dir, 
                                               'darts' if model_name in global_models else 'pickle')
                print(f"\n⏭️  Skipping {model_name} iter{iteration} (already exists: {existing_path})")
                print(f"   Set overwrite.prediction=true in config or use --force to retrain")
                continue
            
            print(f"\n🔧 Training {model_name} (iteration {iteration}/{model_iterations}, seed={seed})...")
            
            # Start performance monitoring
            monitor = PerformanceMonitor()
            monitor.start()
            
            try:
                if model_name in global_models:
                    # Darts deep learning model
                    model = train_global_model_darts(model_name, all_series, pred_config, seed)
                    model_path = save_model(model, model_name, iteration, output_dir, 'darts')
                    
                elif model_name == 'xgboost':
                    # XGBoost model
                    model, lag = train_xgboost_model(all_series, pred_config, seed)
                    # Save model and lag info together
                    model_data = {'model': model, 'lag': lag}
                    model_path = os.path.join(output_dir, f"{model_name}_iter{iteration}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data, f)
                
                # Stop monitoring and get metrics
                metrics = monitor.stop()
                
                print(f"   ✓ Saved: {model_path}")
                print(f"   ⏱️ Time: {metrics['time_seconds']:.2f}s")
                
                # Store training metrics
                training_metrics.append({
                    'model': model_name,
                    'iteration': iteration,
                    'seed': seed,
                    'time_seconds': metrics['time_seconds'],
                    'cpu_cores_used': metrics['cpu_cores_used'],
                    'cpu_cores_total': metrics['cpu_cores_total'],
                    'memory_mb': metrics['memory_mb'],
                    'memory_total_mb': metrics['memory_total_mb'],
                    'gpu_percent': metrics.get('gpu_percent'),
                    'gpu_memory_mb': metrics.get('gpu_memory_mb'),
                    'gpu_memory_total_mb': metrics.get('gpu_memory_total_mb'),
                    'model_path': model_path
                })
                
            except Exception as e:
                monitor.stop()
                print(f"   ❌ Failed to train {model_name}: {e}")
                training_metrics.append({
                    'model': model_name,
                    'iteration': iteration,
                    'seed': seed,
                    'time_seconds': None,
                    'error': str(e)
                })
    
    # Save training metrics
    metrics_filename = generate_training_metrics_filename()
    metrics_path = os.path.join(metrics_dir, metrics_filename)
    
    df_metrics = pd.DataFrame(training_metrics)
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Models saved to: {output_dir}/")
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Summary
    print(f"\n📊 Summary:")
    if len(df_metrics) > 0 and 'time_seconds' in df_metrics.columns:
        successful = df_metrics[df_metrics['time_seconds'].notna()]
        print(f"   Total trained: {len(successful)}/{len(df_metrics)}")
        if len(successful) > 0:
            print(f"   Total time: {successful['time_seconds'].sum():.2f}s")
            print(f"   Avg time per model: {successful['time_seconds'].mean():.2f}s")
    else:
        print(f"   No models were trained (all models may have been skipped or already exist)")
    print("="*70)


if __name__ == "__main__":
    main()

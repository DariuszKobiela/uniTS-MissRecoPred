#!/usr/bin/env python3
"""
Dataset Prediction Script
Predicts future values (test set) using training data.

This script implements TWO training strategies:
1. GLOBAL TRAINING (deep learning models): Train ONE model on ALL data, then predict
2. PER-FILE TRAINING (statistical models): Train separately for each file

For NON-DETERMINISTIC models:
- Training is repeated N times (configurable in prediction_models_config.yaml)
- Each iteration uses a different random seed (base_seed + iteration)
- All predictions are saved with iteration number for statistical analysis

For DETERMINISTIC models (SARIMAX, Holt-Winters, Prophet):
- Training is done only ONCE (they always produce the same result)

Uses config/config.yaml and config/prediction_models_config.yaml for configuration.
Collects performance metrics (time, CPU, RAM, GPU usage).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import prediction models and config loader
from prediction_models import PREDICTION_MODELS, is_gpu_model
from utils.config_loader import load_config, load_prediction_models_config
from utils.performance_metrics import PerformanceMonitor, format_metrics


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        
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


# =============================================================================
# GLOBAL TRAINING FUNCTIONS (for deep learning models)
# =============================================================================

def train_global_model_darts(model_name: str,
                              all_series: List[pd.Series],
                              pred_config,
                              seed: int = None):
    """
    Train a global Darts model on all time series.
    
    Args:
        model_name: Name of the model (lstm, gru, tcn, nbeats, deepar, transformer)
        all_series: List of all training time series
        pred_config: PredictionModelsConfig object
        seed: Random seed for this training iteration
        
    Returns:
        Trained Darts model
    """
    from darts import TimeSeries
    from darts.models import RNNModel, TCNModel, NBEATSModel, TFTModel
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
        # Create datetime index
        date_index = pd.date_range(start='2000-01-01', periods=len(series), freq='H')
        ts = TimeSeries.from_times_and_values(times=date_index, values=series.values, freq='H')
        
        # Split into train/val
        split_point = int(len(ts) * (1 - val_split))
        train_series_list.append(ts[:split_point])
        val_series_list.append(ts[split_point:])
    
    # Setup early stopping callback
    callbacks = []
    if es_enabled:
        early_stopper = EarlyStopping(
            "val_loss",
            patience=es_patience,
            min_delta=es_min_delta,
            verbose=es_verbose
        )
        callbacks.append(early_stopper)
    
    # Create model based on type
    input_chunk_length = model_params.get('input_chunk_length', 100)
    
    # Adjust input_chunk_length if series are too short
    min_series_len = min(len(ts) for ts in train_series_list)
    if min_series_len < input_chunk_length:
        input_chunk_length = max(10, min_series_len // 2)
    
    pl_trainer_kwargs = {
        "callbacks": callbacks,
        "accelerator": "auto",
        "enable_progress_bar": False,
        "enable_model_summary": False
    }
    
    if model_name in ['lstm', 'gru']:
        model = RNNModel(
            model=model_name.upper(),
            input_chunk_length=input_chunk_length,
            training_length=model_params.get('training_length', 24),
            hidden_dim=model_params.get('hidden_dim', 32),
            n_rnn_layers=model_params.get('n_layers', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False
        )
    elif model_name == 'deepar':
        model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            training_length=model_params.get('training_length', 24),
            hidden_dim=model_params.get('hidden_dim', 40),
            n_rnn_layers=model_params.get('n_layers', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            likelihood=GaussianLikelihood(),
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False
        )
    elif model_name == 'tcn':
        output_chunk_length = min(model_params.get('output_chunk_length', 10), min_series_len // 4)
        model = TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            kernel_size=model_params.get('kernel_size', 3),
            num_filters=model_params.get('num_filters', 64),
            dilation_base=model_params.get('dilation_base', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False
        )
    elif model_name == 'nbeats':
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        model = NBEATSModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=output_chunk_length,
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
            save_checkpoints=False
        )
    elif model_name in ['transformer', 'tft']:
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        model = TFTModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=output_chunk_length,
            hidden_size=model_params.get('hidden_size', 64),
            lstm_layers=model_params.get('lstm_layers', 1),
            num_attention_heads=model_params.get('num_attention_heads', 4),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=max_epochs,
            add_relative_index=model_params.get('add_relative_index', True),
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False
        )
    else:
        raise ValueError(f"Unknown global training model: {model_name}")
    
    # Train the model on all series
    model.fit(train_series_list, val_series=val_series_list, verbose=False)
    
    return model


def predict_with_global_model(model, train_series: pd.Series, horizon: int) -> pd.Series:
    """
    Use a trained global model to predict future values.
    
    Args:
        model: Trained Darts model
        train_series: Training time series to predict from
        horizon: Number of steps to predict
        
    Returns:
        Predicted values as pd.Series
    """
    from darts import TimeSeries
    
    # Convert to Darts TimeSeries
    date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
    ts = TimeSeries.from_times_and_values(times=date_index, values=train_series.values, freq='H')
    
    # Predict
    prediction = model.predict(n=horizon, series=ts)
    
    # Convert back to pd.Series
    forecast_values = prediction.values().flatten()
    last_idx = train_series.index[-1]
    forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
    
    return pd.Series(forecast_values, index=forecast_index, name='predicted')


def train_global_xgboost(all_series: List[pd.Series], pred_config, seed: int = None):
    """
    Train a global XGBoost model on all time series using lag features.
    
    Args:
        all_series: List of all training time series
        pred_config: PredictionModelsConfig object
        seed: Random seed for this training iteration
        
    Returns:
        Tuple of (trained model, lag value)
    """
    import xgboost as xgb
    
    model_params = pred_config.get_xgboost_params()
    lag = model_params.get('lags', 10)
    
    # Create combined lag features from all series
    all_X = []
    all_y = []
    
    for series in all_series:
        values = series.values
        for i in range(lag, len(values)):
            features = values[i-lag:i]
            target = values[i]
            all_X.append(features)
            all_y.append(target)
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=model_params.get('n_estimators', 100),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.1),
        random_state=seed,
        verbosity=0
    )
    model.fit(X, y)
    
    return model, lag


def predict_with_global_xgboost(model, train_series: pd.Series, horizon: int, lag: int) -> pd.Series:
    """
    Use a trained global XGBoost model to predict future values recursively.
    
    Args:
        model: Trained XGBoost model
        train_series: Training time series to predict from
        horizon: Number of steps to predict
        lag: Number of lag features
        
    Returns:
        Predicted values as pd.Series
    """
    history = list(train_series.values)
    predictions = []
    
    for _ in range(horizon):
        # Create input features from most recent 'lag' values
        input_features = np.array(history[-lag:]).reshape(1, -1)
        
        # Predict next step
        pred = model.predict(input_features)[0]
        predictions.append(pred)
        
        # Add prediction to history for next step
        history.append(pred)
    
    # Return forecast with integer index
    last_idx = train_series.index[-1]
    forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
    
    return pd.Series(predictions, index=forecast_index, name='predicted')


# =============================================================================
# PER-FILE TRAINING FUNCTIONS (for statistical models)
# =============================================================================

def predict_per_file_statistical(model_name: str,
                                  train_series: pd.Series,
                                  horizon: int,
                                  pred_config,
                                  seed: int = None) -> pd.Series:
    """
    Train a statistical model per-file and predict.
    
    Args:
        model_name: Name of the model (sarimax, holt_winters, prophet)
        train_series: Training time series
        horizon: Number of steps to predict
        pred_config: PredictionModelsConfig object
        seed: Random seed (ignored for deterministic models)
        
    Returns:
        Predicted values as pd.Series
    """
    model_params = pred_config.get_model_params(model_name)
    
    if model_name == 'sarimax':
        from prediction_models.sarimax import predict_sarimax
        order = tuple(model_params.get('order', [1, 1, 1]))
        seasonal_order = tuple(model_params.get('seasonal_order', [1, 1, 1, 12]))
        return predict_sarimax(train_series, horizon, order=order, seasonal_order=seasonal_order)
    
    elif model_name == 'holt_winters':
        from prediction_models.holt_winters import predict_holt_winters
        return predict_holt_winters(
            train_series, horizon,
            seasonal_periods=model_params.get('seasonal_periods', 168),
            trend=model_params.get('trend', 'add'),
            seasonal=model_params.get('seasonal', 'add')
        )
    
    elif model_name == 'prophet':
        from prediction_models.prophet import predict_prophet
        return predict_prophet(
            train_series, horizon,
            yearly_seasonality=model_params.get('yearly_seasonality', False),
            weekly_seasonality=model_params.get('weekly_seasonality', False),
            daily_seasonality=model_params.get('daily_seasonality', False)
        )
    
    else:
        raise ValueError(f"Unknown per-file model: {model_name}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict future values using training data with various prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config/config.yaml
  python 7_predict_datasets.py
  
  # Override with specific models
  python 7_predict_datasets.py --models holt_winters prophet lstm
  
  # Predict only on reconstructed data
  python 7_predict_datasets.py --no-original
  
  # Use custom config file
  python 7_predict_datasets.py --config config/my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--pred-config',
        type=str,
        default='config/prediction_models_config.yaml',
        help='Path to prediction models config (default: config/prediction_models_config.yaml)'
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
    
    parser.add_argument(
        '--iterations',
        type=int,
        help='Override number of training iterations (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    try:
        config = load_config(args.config)
        pred_config = load_prediction_models_config(args.pred_config)
        print(f"✓ Loaded configuration from: {args.config}")
        print(f"✓ Loaded prediction models config from: {args.pred_config}\n")
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
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
    
    # Get number of training iterations
    n_iterations = args.iterations if args.iterations else pred_config.get_training_iterations()
    base_seed = pred_config.get_seed()
    
    # Get horizon (number of steps to predict) from test data
    test_files = list(Path(test_dir).glob("*.csv"))
    if not test_files:
        print(f"❌ No test data found in {test_dir}")
        print("   Run 2_create_split.py first to create train/test split")
        return
    
    # Determine horizon from first test file
    test_df = load_dataset(str(test_files[0]))
    horizon = len(test_df)
    print(f"📊 Prediction horizon: {horizon} steps (from test set size)")
    
    # Get overwrite setting
    overwrite = args.force if args.force else config.get_overwrite_existing()
    
    # Separate models by training type
    global_models = [m for m in models if pred_config.is_global_training_model(m)]
    per_file_models = [m for m in models if pred_config.is_per_file_training_model(m)]
    ml_models = [m for m in models if m in pred_config.get_ml_models()]
    deterministic_models = [m for m in models if pred_config.is_deterministic_model(m)]
    non_deterministic_models = [m for m in models if pred_config.is_non_deterministic_model(m)]
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET PREDICTION")
    print("="*70)
    print(f"Global training models: {global_models}")
    print(f"Per-file training models: {per_file_models}")
    print(f"ML models: {ml_models}")
    print(f"Non-deterministic models: {non_deterministic_models} -> {n_iterations} iterations")
    print(f"Deterministic models: {deterministic_models} -> 1 iteration")
    print(f"Horizon: {horizon} steps")
    print(f"Output directory: {output_dir}")
    print(f"Predict on original training: {predict_on_original}")
    print(f"Predict on reconstructed: {predict_on_reconstructed}")
    print("="*70)
    
    # Collect ALL training data
    all_train_files = []
    all_train_metadata = []
    
    if predict_on_original:
        print(f"\n📋 Collecting original training data from {train_dir}...")
        train_files = list(Path(train_dir).glob("*.csv"))
        for f in train_files:
            all_train_files.append(str(f))
            all_train_metadata.append({
                'file_path': str(f),
                'dataset': f.stem,
                'source_type': 'original',
                'technique': None,
                'rate_percent': None,
                'iteration': None,
                'reconstruction_model': None
            })
        print(f"   Found {len(train_files)} original training files")
    
    if predict_on_reconstructed:
        print(f"\n📋 Collecting reconstructed data from {fixed_dir}...")
        recon_files = list(Path(fixed_dir).glob("*.csv"))
        for f in recon_files:
            try:
                metadata = parse_reconstructed_filename(f.name)
                all_train_files.append(str(f))
                all_train_metadata.append({
                    'file_path': str(f),
                    'dataset': metadata['dataset'],
                    'source_type': 'reconstructed',
                    'technique': metadata['technique'],
                    'rate_percent': metadata['rate_percent'],
                    'iteration': metadata['iteration'],
                    'reconstruction_model': metadata['reconstruction_model']
                })
            except Exception as e:
                print(f"   ⚠️ Could not parse filename {f.name}: {e}")
        print(f"   Found {len(recon_files)} reconstructed files")
    
    if not all_train_files:
        print("❌ No training data found")
        return
    
    print(f"\n📊 Total training files: {len(all_train_files)}")
    
    # Load all training series
    print("\n📂 Loading all training series...")
    all_series = []
    for file_path in tqdm(all_train_files, desc="Loading"):
        df = load_dataset(file_path)
        series = df.iloc[:, 0]
        all_series.append(series)
    
    # Track results
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ==========================================================================
    # NON-DETERMINISTIC MODELS (N iterations)
    # ==========================================================================
    
    for iteration in range(1, n_iterations + 1):
        iteration_seed = base_seed + iteration
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{n_iterations} (seed={iteration_seed})")
        print(f"{'='*70}")
        
        # Train global Darts models for this iteration
        trained_global_models = {}
        
        for model_name in global_models:
            if model_name not in non_deterministic_models:
                continue
                
            try:
                print(f"\n🔧 Training global {model_name.upper()} model (iteration {iteration})...")
                
                monitor = PerformanceMonitor()
                monitor.start()
                
                trained_model = train_global_model_darts(
                    model_name, all_series, pred_config, iteration_seed
                )
                
                training_metrics = monitor.stop()
                print(f"   ✅ Training completed in {training_metrics.get('time_seconds', 0):.1f}s")
                
                trained_global_models[model_name] = {
                    'model': trained_model,
                    'training_metrics': training_metrics
                }
                
            except Exception as e:
                print(f"   ❌ Failed to train global {model_name}: {e}")
                continue
        
        # Train global XGBoost for this iteration
        trained_xgboost = None
        xgboost_lag = None
        
        if 'xgboost' in ml_models and 'xgboost' in non_deterministic_models:
            try:
                print(f"\n🔧 Training global XGBoost model (iteration {iteration})...")
                
                monitor = PerformanceMonitor()
                monitor.start()
                
                trained_xgboost, xgboost_lag = train_global_xgboost(
                    all_series, pred_config, iteration_seed
                )
                
                xgboost_training_metrics = monitor.stop()
                print(f"   ✅ Training completed in {xgboost_training_metrics.get('time_seconds', 0):.1f}s")
                
            except Exception as e:
                print(f"   ❌ Failed to train global XGBoost: {e}")
        
        # Predict using global models for this iteration
        for model_name, model_info in trained_global_models.items():
            print(f"\n🔮 Predicting with global {model_name.upper()} (iteration {iteration})...")
            model = model_info['model']
            
            for i, (file_path, metadata) in enumerate(tqdm(
                zip(all_train_files, all_train_metadata), 
                total=len(all_train_files),
                desc=f"   {model_name}"
            )):
                series = all_series[i]
                
                # Generate output filename with iteration
                if metadata['source_type'] == 'original':
                    output_filename = f"{metadata['dataset']}_original_{model_name}_iter{iteration}.csv"
                else:
                    base_name = Path(file_path).stem
                    output_filename = f"{base_name}_{model_name}_iter{iteration}.csv"
                
                output_file = os.path.join(predictions_dir, output_filename)
                
                # Skip if exists and not overwriting
                if Path(output_file).exists() and not overwrite:
                    all_results.append({
                        'status': 'skipped',
                        'model': model_name,
                        'iteration': iteration,
                        'metadata': metadata
                    })
                    continue
                
                try:
                    monitor = PerformanceMonitor()
                    monitor.start()
                    
                    predictions = predict_with_global_model(model, series, horizon)
                    
                    pred_metrics = monitor.stop()
                    
                    # Save predictions with iteration column
                    output_df = pd.DataFrame({
                        'predicted': predictions.values,
                        'iteration': iteration
                    }, index=predictions.index)
                    output_df.index.name = 'index'
                    output_df.to_csv(output_file)
                    
                    all_results.append({
                        'status': 'success',
                        'model': model_name,
                        'iteration': iteration,
                        'metadata': metadata,
                        'metrics': pred_metrics,
                        'output_file': output_file
                    })
                    
                except Exception as e:
                    all_results.append({
                        'status': 'error',
                        'model': model_name,
                        'iteration': iteration,
                        'metadata': metadata,
                        'message': str(e)
                    })
        
        # Predict using global XGBoost for this iteration
        if trained_xgboost is not None:
            print(f"\n🔮 Predicting with global XGBoost (iteration {iteration})...")
            
            for i, (file_path, metadata) in enumerate(tqdm(
                zip(all_train_files, all_train_metadata),
                total=len(all_train_files),
                desc="   xgboost"
            )):
                series = all_series[i]
                
                # Generate output filename with iteration
                if metadata['source_type'] == 'original':
                    output_filename = f"{metadata['dataset']}_original_xgboost_iter{iteration}.csv"
                else:
                    base_name = Path(file_path).stem
                    output_filename = f"{base_name}_xgboost_iter{iteration}.csv"
                
                output_file = os.path.join(predictions_dir, output_filename)
                
                # Skip if exists
                if Path(output_file).exists() and not overwrite:
                    all_results.append({
                        'status': 'skipped',
                        'model': 'xgboost',
                        'iteration': iteration,
                        'metadata': metadata
                    })
                    continue
                
                try:
                    monitor = PerformanceMonitor()
                    monitor.start()
                    
                    predictions = predict_with_global_xgboost(
                        trained_xgboost, series, horizon, xgboost_lag
                    )
                    
                    pred_metrics = monitor.stop()
                    
                    # Save predictions with iteration column
                    output_df = pd.DataFrame({
                        'predicted': predictions.values,
                        'iteration': iteration
                    }, index=predictions.index)
                    output_df.index.name = 'index'
                    output_df.to_csv(output_file)
                    
                    all_results.append({
                        'status': 'success',
                        'model': 'xgboost',
                        'iteration': iteration,
                        'metadata': metadata,
                        'metrics': pred_metrics,
                        'output_file': output_file
                    })
                    
                except Exception as e:
                    all_results.append({
                        'status': 'error',
                        'model': 'xgboost',
                        'iteration': iteration,
                        'metadata': metadata,
                        'message': str(e)
                    })
    
    # ==========================================================================
    # DETERMINISTIC MODELS (only 1 iteration)
    # ==========================================================================
    
    if deterministic_models:
        print(f"\n{'='*70}")
        print("DETERMINISTIC MODELS (single iteration)")
        print(f"{'='*70}")
    
    for model_name in per_file_models:
        if model_name not in deterministic_models:
            continue
            
        print(f"\n🔮 Training and predicting with {model_name.upper()} (per-file, deterministic)...")
        
        for i, (file_path, metadata) in enumerate(tqdm(
            zip(all_train_files, all_train_metadata),
            total=len(all_train_files),
            desc=f"   {model_name}"
        )):
            series = all_series[i]
            
            # Generate output filename (no iteration suffix for deterministic)
            if metadata['source_type'] == 'original':
                output_filename = f"{metadata['dataset']}_original_{model_name}.csv"
            else:
                base_name = Path(file_path).stem
                output_filename = f"{base_name}_{model_name}.csv"
            
            output_file = os.path.join(predictions_dir, output_filename)
            
            # Skip if exists
            if Path(output_file).exists() and not overwrite:
                all_results.append({
                    'status': 'skipped',
                    'model': model_name,
                    'iteration': 1,
                    'metadata': metadata
                })
                continue
            
            try:
                monitor = PerformanceMonitor()
                monitor.start()
                
                predictions = predict_per_file_statistical(
                    model_name, series, horizon, pred_config, base_seed
                )
                
                pred_metrics = monitor.stop()
                
                # Save predictions (iteration=1 for deterministic)
                output_df = pd.DataFrame({
                    'predicted': predictions.values,
                    'iteration': 1
                }, index=predictions.index)
                output_df.index.name = 'index'
                output_df.to_csv(output_file)
                
                all_results.append({
                    'status': 'success',
                    'model': model_name,
                    'iteration': 1,
                    'metadata': metadata,
                    'metrics': pred_metrics,
                    'output_file': output_file
                })
                
            except Exception as e:
                all_results.append({
                    'status': 'error',
                    'model': model_name,
                    'iteration': 1,
                    'metadata': metadata,
                    'message': str(e)
                })
    
    # ==========================================================================
    # SAVE METRICS AND SUMMARY
    # ==========================================================================
    
    # Count results
    completed = sum(1 for r in all_results if r['status'] == 'success')
    skipped = sum(1 for r in all_results if r['status'] == 'skipped')
    errors = sum(1 for r in all_results if r['status'] == 'error')
    
    # Print errors
    if errors > 0:
        print("\n❌ Errors occurred:")
        error_results = [r for r in all_results if r['status'] == 'error']
        for r in error_results[:10]:
            print(f"  - {r['model']} (iter {r.get('iteration', '?')}): {r.get('message', 'Unknown error')}")
        if len(error_results) > 10:
            print(f"  ... and {len(error_results) - 10} more errors")
    
    # Save performance metrics
    print("\n💾 Saving performance metrics...")
    perf_metrics_dir = os.path.join(output_dir, "performance_metrics")
    os.makedirs(perf_metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(perf_metrics_dir, f"prediction_metrics_{timestamp}.csv")
    
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
                'reconstruction_iteration': metadata.get('iteration'),
                'reconstruction_model': metadata.get('reconstruction_model'),
                'prediction_model': result['model'],
                'prediction_iteration': result.get('iteration', 1),
                'time_seconds': metrics.get('time_seconds', 0),
                'cpu_cores_used': metrics.get('cpu_cores_used', 0),
                'memory_mb': metrics.get('memory_mb', 0),
                'timestamp': timestamp
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"   ✓ Saved {len(metrics_data)} performance records to: {metrics_file}")
    
    # Summary
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"✅ Completed: {completed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Predictions directory: {predictions_dir}")
    print(f"🔄 Training iterations: {n_iterations} (for non-deterministic models)")
    print("="*70)


if __name__ == "__main__":
    main()

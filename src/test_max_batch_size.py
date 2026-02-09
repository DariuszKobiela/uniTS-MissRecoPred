#!/usr/bin/env python3
"""
Test Maximum Batch Size Script
Empirically determines the maximum safe batch_size for each prediction model
by running 1 epoch with progressively larger batch sizes.

Reports:
- RAM usage (system memory)
- VRAM usage (GPU memory)
- Whether the batch_size caused OOM or other errors
- Recommended maximum batch_size per model

Usage:
  # Test all models with default batch sizes (8k, 16k, 32k, 48k, 64k)
  python test_max_batch_size.py

  # Test specific models
  python test_max_batch_size.py --models lstm nbeats

  # Test with custom batch sizes
  python test_max_batch_size.py --batch-sizes 8000 16000 32000 64000 128000

  # Quick test (only 2 batch sizes)
  python test_max_batch_size.py --batch-sizes 8000 32000

  # Limit to fewer training series (faster testing)
  python test_max_batch_size.py --max-series 100
"""

import os
import sys
import warnings

# Suppress NVML warnings from PyTorch before any torch imports
warnings.filterwarnings('ignore', message='.*NVML.*')
warnings.filterwarnings('ignore', message=".*Can't initialize.*")

import argparse
import gc
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
torch.set_float32_matmul_precision('medium')

import psutil

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config, load_prediction_models_config


# =============================================================================
# RESOURCE MONITORING
# =============================================================================

def get_system_memory_info() -> Dict[str, float]:
    """Get current system RAM usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    return {
        'process_rss_mb': mem_info.rss / (1024 * 1024),
        'process_vms_mb': mem_info.vms / (1024 * 1024),
        'system_used_mb': vm.used / (1024 * 1024),
        'system_total_mb': vm.total / (1024 * 1024),
        'system_available_mb': vm.available / (1024 * 1024),
        'system_percent': vm.percent,
    }


def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    """Get current GPU VRAM usage in MB."""
    if not torch.cuda.is_available():
        return None
    
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        return {
            'gpu_name': gpu_props.name,
            'allocated_mb': torch.cuda.memory_allocated(0) / (1024 * 1024),
            'reserved_mb': torch.cuda.memory_reserved(0) / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated(0) / (1024 * 1024),
            'total_mb': gpu_props.total_mem / (1024 * 1024),
        }
    except Exception:
        return None


def reset_gpu_memory_stats():
    """Reset GPU peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.empty_cache()


def format_memory_report(ram: Dict, gpu: Optional[Dict]) -> str:
    """Format memory report for display."""
    lines = []
    lines.append(f"  RAM: Process RSS = {ram['process_rss_mb']:.0f} MB "
                 f"({ram['process_rss_mb']/ram['system_total_mb']*100:.1f}% of {ram['system_total_mb']:.0f} MB)")
    lines.append(f"  RAM: System used = {ram['system_used_mb']:.0f} MB "
                 f"({ram['system_percent']:.1f}%), available = {ram['system_available_mb']:.0f} MB")
    
    if gpu:
        lines.append(f"  GPU: {gpu['gpu_name']}")
        lines.append(f"  GPU: Allocated = {gpu['allocated_mb']:.0f} MB, "
                     f"Peak = {gpu['max_allocated_mb']:.0f} MB, "
                     f"Total = {gpu['total_mb']:.0f} MB "
                     f"({gpu['max_allocated_mb']/gpu['total_mb']*100:.1f}%)")
    
    return "\n".join(lines)


# =============================================================================
# DATA LOADING (mirrors 7_train_prediction_models.py)
# =============================================================================

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
    """Convert pandas Series to Darts TimeSeries (using float32)."""
    from darts import TimeSeries
    
    date_index = pd.date_range(start='2000-01-01', periods=len(series), freq='h')
    ts = TimeSeries.from_times_and_values(
        times=date_index, values=series.values.astype(np.float32), freq='h'
    )
    return ts


def _load_series_from_file(file_path: str) -> Optional[pd.Series]:
    """Load a single CSV and extract its first column. Returns None on failure."""
    try:
        df = load_dataset(file_path)
        series = df.iloc[:, 0].dropna()
        if len(series) > 10:
            return series
    except Exception:
        pass
    return None


def load_all_series(config, max_series: Optional[int] = None) -> List[pd.Series]:
    """Load all training series from config directories (parallel I/O)."""
    train_dir = config.get_splitted_train_dir()
    fixed_dir = config.get_fixed_dir()
    
    csv_files = []
    
    if config.get_predict_on_original_train() and os.path.exists(train_dir):
        csv_files.extend([str(f) for f in sorted(Path(train_dir).glob("*.csv"))])
    
    if config.get_predict_on_reconstructed() and os.path.exists(fixed_dir):
        csv_files.extend([str(f) for f in sorted(Path(fixed_dir).glob("*.csv"))])
    
    if not csv_files:
        return []
    
    # Parallel CSV loading (I/O-bound)
    n_threads = min(16, max(1, os.cpu_count() or 1))
    print(f"   Loading {len(csv_files)} CSV files ({n_threads} threads)...")
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(tqdm(
            executor.map(_load_series_from_file, csv_files),
            total=len(csv_files),
            desc="   Loading CSVs"
        ))
    
    all_series = [s for s in results if s is not None]
    
    # Limit number of series for faster testing
    if max_series and len(all_series) > max_series:
        print(f"   Limiting to {max_series} series (out of {len(all_series)}) for faster testing")
        all_series = all_series[:max_series]
    
    return all_series


def prepare_darts_data(all_series: List[pd.Series], val_split: float):
    """Pre-convert to Darts TimeSeries with train/val split."""
    train_series_list = []
    val_series_list = []
    
    for series in tqdm(all_series, desc="   Converting to Darts"):
        ts = series_to_darts(series)
        
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


# =============================================================================
# MODEL CREATION (mirrors 7_train_prediction_models.py but with 1 epoch)
# =============================================================================

def create_model(model_name: str, batch_size: int, model_params: Dict,
                 min_series_len: int, seed: int = 42):
    """Create a Darts model with given batch_size and 1 epoch (for testing)."""
    from darts.models import RNNModel, TCNModel, NBEATSModel, TFTModel, TransformerModel
    from darts.utils.likelihood_models import GaussianLikelihood
    
    input_chunk_length = model_params.get('input_chunk_length', 24)
    if min_series_len < input_chunk_length:
        input_chunk_length = max(10, min_series_len // 2)
    
    pl_trainer_kwargs = {
        "accelerator": "auto",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "max_epochs": 1,  # Only 1 epoch for testing
    }
    
    if model_name in ['lstm', 'gru']:
        return RNNModel(
            model=model_name.upper(),
            input_chunk_length=input_chunk_length,
            training_length=max(model_params.get('training_length', 30), input_chunk_length),
            hidden_dim=model_params.get('hidden_dim', 32),
            n_rnn_layers=model_params.get('n_layers', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=1,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    elif model_name == 'deepar':
        return RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            training_length=max(model_params.get('training_length', 30), input_chunk_length),
            hidden_dim=model_params.get('hidden_dim', 40),
            n_rnn_layers=model_params.get('n_layers', 2),
            dropout=model_params.get('dropout', 0.1),
            likelihood=GaussianLikelihood(),
            batch_size=batch_size,
            n_epochs=1,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    elif model_name == 'tcn':
        output_chunk_length = min(model_params.get('output_chunk_length', 6), min_series_len // 4)
        return TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=max(1, output_chunk_length),
            kernel_size=model_params.get('kernel_size', 3),
            num_filters=model_params.get('num_filters', 64),
            dilation_base=model_params.get('dilation_base', 2),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=1,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    elif model_name == 'nbeats':
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        return NBEATSModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=max(1, output_chunk_length),
            generic_architecture=model_params.get('generic_architecture', True),
            num_stacks=model_params.get('num_stacks', 30),
            num_blocks=model_params.get('num_blocks', 1),
            num_layers=model_params.get('num_layers', 4),
            layer_widths=model_params.get('layer_widths', 256),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=1,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    elif model_name == 'vanilla_transformer':
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        return TransformerModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=max(1, output_chunk_length),
            d_model=model_params.get('d_model', 64),
            nhead=model_params.get('nhead', 4),
            num_encoder_layers=model_params.get('num_encoder_layers', 2),
            num_decoder_layers=model_params.get('num_decoder_layers', 2),
            dim_feedforward=model_params.get('dim_feedforward', 128),
            dropout=model_params.get('dropout', 0.1),
            batch_size=batch_size,
            n_epochs=1,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    elif model_name == 'temporal_fusion_transformer':
        output_chunk_length = min(model_params.get('output_chunk_length', 12), min_series_len // 4)
        return TFTModel(
            input_chunk_length=model_params.get('input_chunk_length', 24),
            output_chunk_length=max(1, output_chunk_length),
            hidden_size=model_params.get('hidden_size', 64),
            lstm_layers=model_params.get('lstm_layers', 1),
            num_attention_heads=model_params.get('num_attention_heads', 4),
            dropout=model_params.get('dropout', 0.1),
            add_relative_index=model_params.get('add_relative_index', True),
            batch_size=batch_size,
            n_epochs=1,
            random_state=seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            force_reset=True,
            save_checkpoints=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# BATCH SIZE TESTING
# =============================================================================

def test_batch_size(model_name: str, batch_size: int, model_params: Dict,
                    train_series: List, val_series: List,
                    min_series_len: int,
                    dataloader_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Test a single batch_size for a model by running 1 epoch.
    
    Returns:
        Dict with results: success, ram, gpu, time, error
    """
    result = {
        'model': model_name,
        'batch_size': batch_size,
        'success': False,
        'time_seconds': None,
        'ram_before_mb': None,
        'ram_after_mb': None,
        'ram_peak_mb': None,
        'gpu_peak_mb': None,
        'gpu_total_mb': None,
        'gpu_peak_percent': None,
        'error': None,
    }
    
    # Record RAM before
    ram_before = get_system_memory_info()
    result['ram_before_mb'] = ram_before['process_rss_mb']
    
    # Reset GPU stats
    reset_gpu_memory_stats()
    
    try:
        start_time = time.time()
        
        # Create and train model for 1 epoch
        model = create_model(model_name, batch_size, model_params, min_series_len)
        
        fit_kwargs = {"verbose": False}
        if dataloader_kwargs:
            fit_kwargs["dataloader_kwargs"] = dataloader_kwargs
        
        if val_series:
            model.fit(train_series, val_series=val_series, **fit_kwargs)
        else:
            model.fit(train_series, **fit_kwargs)
        
        elapsed = time.time() - start_time
        
        # Record RAM after
        ram_after = get_system_memory_info()
        result['ram_after_mb'] = ram_after['process_rss_mb']
        result['ram_peak_mb'] = ram_after['process_rss_mb']  # Approximation
        
        # Record GPU
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            result['gpu_peak_mb'] = gpu_info['max_allocated_mb']
            result['gpu_total_mb'] = gpu_info['total_mb']
            result['gpu_peak_percent'] = (gpu_info['max_allocated_mb'] / gpu_info['total_mb']) * 100
        
        result['success'] = True
        result['time_seconds'] = round(elapsed, 2)
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        error_str = str(e)
        if "out of memory" in error_str.lower() or "CUDA" in error_str:
            result['error'] = "GPU OOM"
        else:
            result['error'] = f"RuntimeError: {error_str[:200]}"
        
        # Still try to get GPU info
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            result['gpu_peak_mb'] = gpu_info['max_allocated_mb']
            result['gpu_total_mb'] = gpu_info['total_mb']
            result['gpu_peak_percent'] = (gpu_info['max_allocated_mb'] / gpu_info['total_mb']) * 100
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except MemoryError:
        result['error'] = "RAM OOM"
        gc.collect()
        
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)[:200]}"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test maximum batch_size for each prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_max_batch_size.py
  python test_max_batch_size.py --models lstm nbeats
  python test_max_batch_size.py --batch-sizes 8000 16000 32000 64000 128000
  python test_max_batch_size.py --max-series 100  # faster testing
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
        help='Specific models to test (default: all global training models)'
    )
    
    parser.add_argument(
        '--batch-sizes',
        nargs='+',
        type=int,
        default=[8000, 16000, 32000, 48000, 64000],
        help='Batch sizes to test (default: 8000 16000 32000 48000 64000)'
    )
    
    parser.add_argument(
        '--max-series',
        type=int,
        default=None,
        help='Limit number of training series (for faster testing)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: batch_size_test_results_<timestamp>.csv)'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # LOAD CONFIG
    # =========================================================================
    try:
        config = load_config(args.config)
        pred_config = load_prediction_models_config()
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return
    
    # =========================================================================
    # DETERMINE MODELS TO TEST
    # =========================================================================
    global_models = pred_config.get_global_training_models()
    
    if args.models:
        models_to_test = [m for m in args.models if m in global_models]
        if not models_to_test:
            print(f"No valid models specified. Available: {global_models}")
            return
    else:
        models_to_test = global_models
    
    batch_sizes = sorted(args.batch_sizes)
    
    # =========================================================================
    # PRINT HEADER
    # =========================================================================
    print("=" * 80)
    print("TEST MAXIMUM BATCH SIZE")
    print("=" * 80)
    print(f"Models to test: {models_to_test}")
    print(f"Batch sizes:    {batch_sizes}")
    print(f"Max series:     {args.max_series or 'all'}")
    print()
    
    # Print system info
    ram = get_system_memory_info()
    gpu = get_gpu_memory_info()
    print("System Resources:")
    print(format_memory_report(ram, gpu))
    print()
    
    # =========================================================================
    # LOAD AND PREPARE DATA
    # =========================================================================
    print("Loading training data...")
    all_series = load_all_series(config, max_series=args.max_series)
    
    if not all_series:
        print("No training data found!")
        return
    
    print(f"Loaded {len(all_series)} series")
    
    # Convert to float32
    print("Converting to float32...")
    for i in range(len(all_series)):
        all_series[i] = all_series[i].astype(np.float32)
    gc.collect()
    
    # Pre-convert to Darts
    print("Pre-converting to Darts TimeSeries...")
    val_split = pred_config.get_validation_split()
    train_series, val_series = prepare_darts_data(all_series, val_split)
    
    # Free raw data
    del all_series
    gc.collect()
    
    print(f"Train series: {len(train_series)}, Val series: {len(val_series)}")
    
    # Get min series length (needed for model creation)
    min_series_len = min(len(ts) for ts in train_series)
    print(f"Min series length: {min_series_len}")
    
    # Show RAM after data loading
    ram = get_system_memory_info()
    print(f"\nRAM after data loading: {ram['process_rss_mb']:.0f} MB "
          f"({ram['system_percent']:.1f}% system used)")
    print()
    
    # =========================================================================
    # RUN TESTS
    # =========================================================================
    all_results = []
    dataloader_kwargs = pred_config.get_dataloader_kwargs()
    if dataloader_kwargs:
        print(f"DataLoader kwargs: {dataloader_kwargs}")
    
    for model_name in models_to_test:
        model_params = pred_config.get_model_params(model_name)
        current_batch_size = pred_config.get_model_batch_size(model_name)
        
        print("=" * 80)
        print(f"TESTING: {model_name.upper()} (current config batch_size={current_batch_size})")
        print("=" * 80)
        
        max_successful_batch = 0
        
        for batch_size in batch_sizes:
            print(f"\n  batch_size={batch_size:,}...", end=" ", flush=True)
            
            result = test_batch_size(
                model_name, batch_size, model_params,
                train_series, val_series, min_series_len,
                dataloader_kwargs=dataloader_kwargs
            )
            all_results.append(result)
            
            if result['success']:
                max_successful_batch = batch_size
                gpu_str = ""
                if result['gpu_peak_mb'] is not None:
                    gpu_str = (f", GPU: {result['gpu_peak_mb']:.0f}/{result['gpu_total_mb']:.0f} MB "
                              f"({result['gpu_peak_percent']:.1f}%)")
                print(f"OK ({result['time_seconds']:.1f}s, "
                      f"RAM: {result['ram_after_mb']:.0f} MB{gpu_str})")
            else:
                print(f"FAILED ({result['error']})")
                
                # If GPU OOM or RAM OOM, skip larger batch sizes for this model
                if result['error'] in ("GPU OOM", "RAM OOM"):
                    print(f"  Skipping larger batch sizes for {model_name} (OOM detected)")
                    # Fill remaining with skipped
                    remaining = [bs for bs in batch_sizes if bs > batch_size]
                    for skip_bs in remaining:
                        all_results.append({
                            'model': model_name,
                            'batch_size': skip_bs,
                            'success': False,
                            'time_seconds': None,
                            'ram_before_mb': None,
                            'ram_after_mb': None,
                            'ram_peak_mb': None,
                            'gpu_peak_mb': None,
                            'gpu_total_mb': None,
                            'gpu_peak_percent': None,
                            'error': f"Skipped (OOM at {batch_size})",
                        })
                    break
        
        print(f"\n  >>> Max successful batch_size for {model_name}: {max_successful_batch:,}")
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Summary table per model
    print(f"\n{'Model':<30} {'Max Safe Batch':>15} {'Current Config':>15} {'Recommendation':>20}")
    print("-" * 80)
    
    for model_name in models_to_test:
        model_results = df_results[df_results['model'] == model_name]
        successful = model_results[model_results['success'] == True]
        
        current_bs = pred_config.get_model_batch_size(model_name)
        
        if len(successful) > 0:
            max_safe = successful['batch_size'].max()
            
            # Recommendation: use 80% of max successful batch_size as safety margin
            recommended = int(max_safe * 0.8)
            # Round to nearest 1000
            recommended = max(1000, (recommended // 1000) * 1000)
            
            print(f"{model_name:<30} {max_safe:>15,} {current_bs:>15,} {recommended:>20,}")
        else:
            print(f"{model_name:<30} {'N/A':>15} {current_bs:>15,} {'reduce batch_size':>20}")
    
    # Detailed results
    print(f"\n\nDetailed Results:")
    print(f"{'Model':<25} {'Batch':>8} {'OK?':>5} {'Time(s)':>8} "
          f"{'RAM(MB)':>10} {'GPU Peak(MB)':>12} {'GPU%':>6} {'Error':>20}")
    print("-" * 100)
    
    for _, row in df_results.iterrows():
        ok = "Yes" if row['success'] else "No"
        time_str = f"{row['time_seconds']:.1f}" if row['time_seconds'] else "-"
        ram_str = f"{row['ram_after_mb']:.0f}" if row['ram_after_mb'] else "-"
        gpu_str = f"{row['gpu_peak_mb']:.0f}" if row['gpu_peak_mb'] else "-"
        gpu_pct = f"{row['gpu_peak_percent']:.1f}" if row['gpu_peak_percent'] else "-"
        err_str = str(row['error'])[:20] if row['error'] else ""
        
        print(f"{row['model']:<25} {row['batch_size']:>8,} {ok:>5} {time_str:>8} "
              f"{ram_str:>10} {gpu_str:>12} {gpu_pct:>6} {err_str:>20}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = "prediction_experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"batch_size_test_results_{timestamp}.csv")
    
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # =========================================================================
    # SUGGESTED CONFIG CHANGES
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUGGESTED CONFIG CHANGES (prediction_models_config.yaml)")
    print("=" * 80)
    
    for model_name in models_to_test:
        model_results = df_results[df_results['model'] == model_name]
        successful = model_results[model_results['success'] == True]
        
        if len(successful) > 0:
            max_safe = successful['batch_size'].max()
            recommended = max(1000, (int(max_safe * 0.8) // 1000) * 1000)
            current_bs = pred_config.get_model_batch_size(model_name)
            
            if recommended != current_bs:
                print(f"  {model_name}:")
                print(f"    batch_size: {recommended}  # was {current_bs}, max tested OK: {max_safe}")
            else:
                print(f"  {model_name}: OK (current={current_bs}, max tested={max_safe})")
        else:
            print(f"  {model_name}: FAILED all batch sizes -- reduce batch_size!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

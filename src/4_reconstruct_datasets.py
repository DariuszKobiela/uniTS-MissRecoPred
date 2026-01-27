#!/usr/bin/env python3
"""
Dataset Reconstruction Script
Repairs degraded training time series datasets using various reconstruction models.
Uses config.yaml for configuration.
Collects performance metrics (time, CPU, RAM, GPU usage).

NOTE: This script operates on degraded TRAINING data only (from data/3_missing_data/).
Test data is preserved separately for prediction evaluation.
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

# Import reconstruction models and config loader
from reconstruction_models import RECONSTRUCTION_MODELS
from utils.config_loader import load_config
from utils.performance_metrics import PerformanceMonitor, format_metrics


def load_degraded_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a degraded dataset.
    
    Args:
        file_path: Path to degraded CSV file
        
    Returns:
        DataFrame with timestamps as index and numeric values
    """
    # Read CSV, treating empty strings as NaN
    df = pd.read_csv(file_path, index_col=0, na_values=['', ' '])
    
    # Convert the first column to numeric (coerce errors to NaN)
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    # Try to parse index as datetime, keep as-is if it fails
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        # If datetime parsing fails, try numeric
        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            # Keep original index if both conversions fail
            pass
    
    return df


def is_gpu_model(model_name: str) -> bool:
    """
    Check if a model requires GPU (Stable Diffusion models).
    
    Args:
        model_name: Name of the reconstruction model
        
    Returns:
        True if model requires GPU, False otherwise
    """
    return model_name.startswith('stable_diffusion_2')


def process_single_reconstruction(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to process a single reconstruction task.
    
    Args:
        task: Dictionary with keys: degraded_file, output_file, model_name, config, force, metadata
        
    Returns:
        Dictionary with keys: status, message, output_file, model, metrics, metadata
    """
    try:
        # Check if file already exists
        if Path(task['output_file']).exists() and not task['force']:
            return {
                'status': 'skipped',
                'message': f"Already exists",
                'output_file': task['output_file'],
                'model': task['model_name'],
                'metadata': task.get('metadata', {}),
                'metrics': None
            }
        
        # Perform reconstruction and collect metrics
        metrics = reconstruct_dataset(
            degraded_file=task['degraded_file'],
            output_file=task['output_file'],
            reconstruction_model=task['model_name'],
            config=task['config']
        )
        
        return {
            'status': 'success',
            'message': 'Completed',
            'output_file': task['output_file'],
            'model': task['model_name'],
            'metadata': task.get('metadata', {}),
            'metrics': metrics
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'output_file': task['output_file'],
            'model': task['model_name'],
            'metadata': task.get('metadata', {}),
            'metrics': None
        }


def reconstruct_dataset(degraded_file: str,
                       output_file: str,
                       reconstruction_model: str,
                       config = None) -> Dict[str, float]:
    """
    Reconstruct a single degraded dataset and collect performance metrics.
    
    Args:
        degraded_file: Path to degraded CSV file
        output_file: Path to output reconstructed CSV file
        reconstruction_model: Name of reconstruction model to use
        config: Configuration object (for SD model settings)
    
    Returns:
        Dictionary with performance metrics (time_seconds, cpu_percent, memory_mb, etc.)
    """
    # Load degraded dataset
    df = load_degraded_dataset(degraded_file)
    series = df.iloc[:, 0]  # First column is the time series
    
    # Get reconstruction model function
    if reconstruction_model not in RECONSTRUCTION_MODELS:
        raise ValueError(f"Unknown reconstruction model: {reconstruction_model}")
    
    model_func = RECONSTRUCTION_MODELS[reconstruction_model]
    
    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Apply reconstruction
    # Check if this is a Stable Diffusion model and pass config parameters
    if reconstruction_model.startswith('stable_diffusion_2') and config:
        sd_settings = config.get_stable_diffusion_settings()
        print(f"    Applying {reconstruction_model}... (steps={sd_settings['num_inference_steps']}, guidance={sd_settings['guidance_scale']})")
        reconstructed_series = model_func(
            series,
            num_inference_steps=sd_settings['num_inference_steps'],
            guidance_scale=sd_settings['guidance_scale']
        )
    else:
        print(f"    Applying {reconstruction_model}...")
        reconstructed_series = model_func(series)
    
    # Stop monitoring and collect metrics
    metrics = monitor.stop()
    
    # Create output DataFrame with original timestamps
    output_df = df.copy()
    output_df.iloc[:, 0] = reconstructed_series.values
    
    # Save reconstructed dataset
    output_df.to_csv(output_file)
    print(f"    ✓ Saved to: {output_file}")
    print(f"    📊 {format_metrics(metrics)}")
    
    return metrics


def parse_degraded_filename(filename: str) -> dict:
    """
    Parse degraded filename to extract metadata.
    Format: datasetName_technique_rateP_iteration.csv
    Dataset name can contain underscores.
    
    Returns:
        dict with keys: dataset, technique, rate_percent, iteration
    """
    base_name = filename.replace('.csv', '')
    parts = base_name.split('_')
    
    if len(parts) < 4:
        raise ValueError(f"Invalid degraded filename format: {filename}")
    
    # Last 3 parts are: technique, rateP, iteration
    # Everything before that is the dataset name
    iteration = int(parts[-1])
    rate_with_p = parts[-2]
    rate_percent = int(rate_with_p.replace('p', ''))
    technique = parts[-3]
    dataset = '_'.join(parts[:-3])  # Join all parts except last 3
    
    return {
        'dataset': dataset,
        'technique': technique,
        'rate_percent': rate_percent,
        'iteration': iteration
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct degraded univariate time series datasets using various reconstruction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config.yaml
  python reconstruct_datasets.py
  
  # Override with specific models
  python reconstruct_datasets.py --models interpolate_linear knn
  
  # Use custom config file
  python reconstruct_datasets.py --config my_config.yaml
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
        choices=list(RECONSTRUCTION_MODELS.keys()) + ['all'],
        help='Reconstruction models to apply (overrides config)'
    )
    
    
    parser.add_argument(
        '--filter-dataset',
        nargs='+',
        help='Only process specific datasets (e.g., boiler pump)'
    )
    
    parser.add_argument(
        '--filter-technique',
        nargs='+',
        help='Only process specific missingness techniques (e.g., MCAR MAR)'
    )
    
    parser.add_argument(
        '--filter-rate',
        nargs='+',
        type=int,
        help='Only process specific missing rates as percentages (e.g., 2 5 10)'
    )
    
    parser.add_argument(
        '--filter-iteration',
        nargs='+',
        type=int,
        help='Only process specific iterations (e.g., 1 2 3)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing reconstructed datasets'
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
    input_dir = config.get_missing_dir()
    output_dir = config.get_fixed_dir()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get models - priority: CLI args > config
    if args.models:
        if 'all' in args.models:
            models = list(RECONSTRUCTION_MODELS.keys())
        else:
            models = args.models
    else:
        models = config.get_reconstruction_models()
    
    if not models:
        print("❌ No reconstruction models specified")
        return
    
    # Get list of degraded files
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    degraded_files = sorted(input_path.glob("*.csv"))
    
    if not degraded_files:
        print(f"❌ No degraded datasets found in {input_dir}")
        return
    
    # Apply filters if provided
    filtered_files = []
    for file in degraded_files:
        try:
            metadata = parse_degraded_filename(file.name)
            
            # Apply CLI filters if provided
            if args.filter_dataset and metadata['dataset'] not in args.filter_dataset:
                continue
            if args.filter_technique and metadata['technique'] not in args.filter_technique:
                continue
            if args.filter_rate and metadata['rate_percent'] not in args.filter_rate:
                continue
            if args.filter_iteration and metadata['iteration'] not in args.filter_iteration:
                continue
            
            filtered_files.append(file)
        except Exception as e:
            print(f"⚠️  Warning: Could not parse filename {file.name}: {e}")
            continue
    
    # Calculate total operations
    total_operations = len(filtered_files) * len(models)
    
    print("="*70)
    print("DATASET RECONSTRUCTION")
    print("="*70)
    print(f"Degraded datasets: {len(filtered_files)}")
    print(f"Reconstruction models ({len(models)}): {models[:5]}{'...' if len(models) > 5 else ''}")
    print(f"Total operations: {total_operations}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if args.filter_dataset:
        print(f"Filtered datasets: {args.filter_dataset}")
    if args.filter_technique:
        print(f"Filtered techniques: {args.filter_technique}")
    if args.filter_rate:
        print(f"Filtered rates: {args.filter_rate}%")
    if args.filter_iteration:
        print(f"Filtered iterations: {args.filter_iteration}")
    print("="*70)
    
    # Build list of all reconstruction tasks
    print("\n📋 Building task list...")
    tasks = []
    overwrite = args.force if args.force else config.get_overwrite_existing()
    
    for degraded_file in filtered_files:
        try:
            metadata = parse_degraded_filename(degraded_file.name)
            
            for model_name in models:
                # Generate output filename
                # Format: datasetName_technique_rateP_iteration_model.csv
                base_name = degraded_file.stem
                output_filename = f"{base_name}_{model_name}.csv"
                output_file = os.path.join(output_dir, output_filename)
                
                tasks.append({
                    'degraded_file': str(degraded_file),
                    'output_file': output_file,
                    'model_name': model_name,
                    'config': config,
                    'force': overwrite,
                    'metadata': metadata  # Include metadata for performance tracking
                })
                
        except Exception as e:
            print(f"❌ Error parsing {degraded_file.name}: {e}")
            continue
    
    # Separate GPU and CPU tasks for smart scheduling
    gpu_tasks = [t for t in tasks if is_gpu_model(t['model_name'])]
    cpu_tasks = [t for t in tasks if not is_gpu_model(t['model_name'])]
    
    n_jobs = config.get_n_jobs()
    print(f"🚀 Processing {len(tasks)} tasks total:")
    print(f"   - {len(cpu_tasks)} CPU model tasks (parallel with {n_jobs} jobs)")
    print(f"   - {len(gpu_tasks)} GPU model tasks (sequential)\n")
    
    all_results = []
    
    # Process CPU models in parallel
    if cpu_tasks:
        print(f"⚡ Processing CPU models in parallel...")
        cpu_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_single_reconstruction)(task) 
            for task in tqdm(cpu_tasks, desc="⏳ CPU models", unit="task", ncols=80)
        )
        all_results.extend(cpu_results)
    
    # Process GPU models sequentially (n_jobs=1)
    if gpu_tasks:
        print(f"\n🎨 Processing GPU models sequentially...")
        gpu_results = Parallel(n_jobs=1, backend='loky')(
            delayed(process_single_reconstruction)(task) 
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
        for r in error_results[:10]:  # Show first 10 errors
            print(f"  - {r['model']}: {r['message']}")
        if len(error_results) > 10:
            print(f"  ... and {len(error_results) - 10} more errors")
    
    # Save performance metrics to CSV (permanent archive + for merging with MAD)
    print("\n💾 Saving performance metrics...")
    results_dir = config.get_results_dir()
    perf_metrics_dir = os.path.join(results_dir, "performance_metrics")
    os.makedirs(perf_metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(perf_metrics_dir, f"performance_metrics_{timestamp}.csv")
    
    # Collect metrics from all successful reconstructions
    metrics_data = []
    for result in all_results:
        if result['status'] == 'success' and result.get('metrics'):
            metadata = result.get('metadata', {})
            metrics = result['metrics']
            
            metrics_data.append({
                'dataset_name': metadata.get('dataset', 'unknown'),
                'technique': metadata.get('technique', 'unknown'),
                'rate_percent': metadata.get('rate_percent', 0),
                'iteration': metadata.get('iteration', 0),
                'model': result['model'],
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
        print("   ⚠️  No performance metrics collected (all files were skipped or failed)")
    
    print("\n" + "="*70)
    print("RECONSTRUCTION COMPLETE")
    print("="*70)
    print(f"✅ Completed: {completed}/{len(tasks)}")
    print(f"⏭️  Skipped (existing): {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Output directory: {output_dir}")
    if metrics_data:
        print(f"📊 Performance metrics: {metrics_file}")
        print(f"   (will be merged with MAD in step 4)")
    print("="*70)


if __name__ == "__main__":
    main()


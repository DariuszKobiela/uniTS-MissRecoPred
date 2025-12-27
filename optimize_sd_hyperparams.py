#!/usr/bin/env python3
"""
Stable Diffusion Hyperparameter Optimization Script

This script tests different combinations of num_inference_steps and guidance_scale
on ALL available degraded datasets and ALL Stable Diffusion models (dynamically 
discovered from reconstruction_models/) to find globally optimal hyperparameters.

The script automatically detects all models that start with 'stable_diffusion_' 
from the reconstruction_models registry, so new SD models will be automatically 
included without code changes.

Usage:
    python optimize_sd_hyperparams.py
    python optimize_sd_hyperparams.py --steps 10 20 50 --guidance 5.0 7.5
    python optimize_sd_hyperparams.py --max-files 5  # Quick test on 5 files only
"""

import os
import sys
import argparse
import time
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import framework modules
from config_loader import load_config
from reconstruction_models import RECONSTRUCTION_MODELS


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Optimize Stable Diffusion hyperparameters for time series reconstruction. "
                    "Tests on ALL available degraded datasets and ALL SD models "
                    "(dynamically discovered from reconstruction_models/) for robust global optimization."
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[5, 10, 20, 30, 50, 75, 100],
        help='List of num_inference_steps to test (default: 5 10 20 30 50)'
    )
    
    parser.add_argument(
        '--guidance',
        type=float,
        nargs='+',
        default=[3.5, 5.0, 7.5, 10.0],
        help='List of guidance_scale values to test (default: 3.5 5.0 7.5 10.0)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Limit number of degraded files to test (for quick testing). Default: all files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='hyperparameter_optimization',
        help='Output directory for results (default: hyperparameter_optimization)'
    )
    
    return parser.parse_args()


def parse_degraded_filename(filename: str) -> dict:
    """
    Parse degraded filename to extract metadata.
    Format: datasetName_technique_rateP_iteration.csv
    """
    name_without_ext = filename.replace('.csv', '')
    parts = name_without_ext.split('_')
    
    if len(parts) < 4:
        return None
    
    # Find the rate pattern (XXp)
    rate_idx = None
    for i, part in enumerate(parts):
        if part.endswith('p') and part[:-1].isdigit():
            rate_idx = i
            break
    
    if rate_idx is None or rate_idx < 1:
        return None
    
    technique = parts[rate_idx - 1]
    rate_percent = int(parts[rate_idx].replace('p', ''))
    iteration = int(parts[rate_idx + 1])
    dataset_name = '_'.join(parts[:rate_idx - 1])
    
    return {
        'dataset': dataset_name,
        'technique': technique,
        'rate_percent': rate_percent,
        'iteration': iteration
    }


def load_degraded_dataset(file_path: str) -> pd.Series:
    """Load degraded dataset"""
    df = pd.read_csv(file_path, index_col=0, na_values=['', ' '])
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            pass
    
    return df.iloc[:, 0]


def load_source_dataset(file_path: str, config) -> pd.Series:
    """Load source dataset"""
    format_settings = config.get_csv_format(os.path.basename(file_path))
    
    df = pd.read_csv(file_path, **format_settings)
    if len(df.columns) >= 2:
        df.set_index(df.columns[0], inplace=True)
    
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            pass
    
    return df.iloc[:, 0]


def calculate_mad(source: pd.Series, degraded: pd.Series, reconstructed: pd.Series) -> dict:
    """Calculate MAD and other metrics for missing values only"""
    # Align indices
    common_index = source.index.intersection(degraded.index).intersection(reconstructed.index)
    source_vals = source.loc[common_index]
    degraded_vals = degraded.loc[common_index]
    reconstructed_vals = reconstructed.loc[common_index]
    
    # Identify missing points
    missing_mask = degraded_vals.isna()
    
    if not missing_mask.any():
        return {'mad': np.nan, 'max_diff': np.nan, 'min_diff': np.nan, 'std_diff': np.nan, 'n_missing': 0}
    
    # Calculate differences for missing points only
    source_missing = source_vals[missing_mask]
    reconstructed_missing = reconstructed_vals[missing_mask]
    
    differences = (source_missing - reconstructed_missing).abs()
    
    return {
        'mad': differences.mean(),
        'max_diff': differences.max(),
        'min_diff': differences.min(),
        'std_diff': differences.std(),
        'n_missing': len(differences)
    }


def get_stable_diffusion_models() -> list:
    """
    Dynamically discover all Stable Diffusion models from RECONSTRUCTION_MODELS.
    Returns list of model names that start with 'stable_diffusion_'.
    """
    sd_models = [
        model_name 
        for model_name in RECONSTRUCTION_MODELS.keys() 
        if model_name.startswith('stable_diffusion_')
    ]
    
    if not sd_models:
        print("⚠️  Warning: No Stable Diffusion models found in reconstruction_models/")
        print("   Make sure models follow naming convention: stable_diffusion_*")
    
    return sorted(sd_models)


def test_configuration(model_name: str, series: pd.Series, num_steps: int, guidance: float) -> dict:
    """Test a single hyperparameter configuration"""
    model_func = RECONSTRUCTION_MODELS[model_name]
    
    # Measure time
    start_time = time.time()
    
    try:
        reconstructed = model_func(series, num_inference_steps=num_steps, guidance_scale=guidance)
        elapsed_time = time.time() - start_time
        status = 'success'
        error = None
    except Exception as e:
        elapsed_time = time.time() - start_time
        reconstructed = series  # Return original on error
        status = 'error'
        error = str(e)
    
    return {
        'reconstructed': reconstructed,
        'time_seconds': elapsed_time,
        'status': status,
        'error': error
    }


def main():
    args = parse_args()
    
    # Dynamically discover all Stable Diffusion models
    SD_MODELS = get_stable_diffusion_models()
    
    if not SD_MODELS:
        print("❌ Error: No Stable Diffusion models found!")
        print("   Make sure you have models named 'stable_diffusion_*' in reconstruction_models/")
        return
    
    print("="*70)
    print("🔬 STABLE DIFFUSION HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print("Testing on ALL available degraded datasets for global optimization")
    print(f"Testing {len(args.steps)} × {len(args.guidance)} = {len(args.steps) * len(args.guidance)} configurations per dataset")
    print(f"Models ({len(SD_MODELS)}): {', '.join(SD_MODELS)}")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Find all degraded files
    missing_dir = Path(config.get_missing_dir())
    if not missing_dir.exists():
        print(f"❌ Error: Missing data directory not found: {missing_dir}")
        print(f"   Run: python 2_degrade_datasets.py first")
        return
    
    degraded_files = sorted(missing_dir.glob("*.csv"))
    
    if not degraded_files:
        print(f"❌ Error: No degraded files found in {missing_dir}")
        print(f"   Run: python 2_degrade_datasets.py first")
        return
    
    # Limit files if requested
    if args.max_files:
        degraded_files = degraded_files[:args.max_files]
        print(f"\n⚠️  Limited to first {args.max_files} files for quick testing")
    
    print(f"\n📂 Found {len(degraded_files)} degraded files to test")
    
    # Parse file metadata and prepare test cases
    test_cases = []
    source_dir = Path(config.get_source_dir())
    
    for degraded_file in degraded_files:
        metadata = parse_degraded_filename(degraded_file.name)
        if not metadata:
            print(f"⚠️  Skipping {degraded_file.name}: Cannot parse filename")
            continue
        
        source_file = source_dir / f"{metadata['dataset']}.csv"
        if not source_file.exists():
            print(f"⚠️  Skipping {degraded_file.name}: Source file not found")
            continue
        
        test_cases.append({
            'degraded_file': degraded_file,
            'source_file': source_file,
            'metadata': metadata
        })
    
    if not test_cases:
        print("❌ No valid test cases found")
        return
    
    print(f"✓ Prepared {len(test_cases)} test cases")
    
    # Display dataset distribution
    datasets = {}
    techniques = {}
    rates = {}
    for case in test_cases:
        m = case['metadata']
        datasets[m['dataset']] = datasets.get(m['dataset'], 0) + 1
        techniques[m['technique']] = techniques.get(m['technique'], 0) + 1
        rates[m['rate_percent']] = rates.get(m['rate_percent'], 0) + 1
    
    print(f"\n📊 Test case distribution:")
    print(f"   Datasets: {len(datasets)} unique ({', '.join(datasets.keys())})")
    print(f"   Techniques: {', '.join(f'{k}={v}' for k, v in techniques.items())}")
    print(f"   Rates: {', '.join(f'{k}%={v}' for k, v in rates.items())}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test all configurations
    results = []
    total_tests = len(test_cases) * len(SD_MODELS) * len(args.steps) * len(args.guidance)
    
    print(f"\n🚀 Running {total_tests} total tests...")
    print(f"   (This may take several hours depending on your hardware)\n")
    
    with tqdm(total=total_tests, desc="Testing configurations", unit="test", ncols=100) as pbar:
        for test_case in test_cases:
            # Load data
            try:
                source_series = load_source_dataset(str(test_case['source_file']), config)
                degraded_series = load_degraded_dataset(str(test_case['degraded_file']))
            except Exception as e:
                print(f"\n⚠️  Error loading {test_case['degraded_file'].name}: {e}")
                pbar.update(len(SD_MODELS) * len(args.steps) * len(args.guidance))
                continue
            
            for model_name in SD_MODELS:
                if model_name not in RECONSTRUCTION_MODELS:
                    print(f"\n⚠️  Warning: Model {model_name} not found in RECONSTRUCTION_MODELS")
                    pbar.update(len(args.steps) * len(args.guidance))
                    continue
                
                for num_steps, guidance in itertools.product(args.steps, args.guidance):
                    pbar.set_description(
                        f"{test_case['metadata']['dataset'][:20]} | "
                        f"{model_name[:25]} | "
                        f"s={num_steps} g={guidance}"
                    )
                    
                    # Test configuration
                    test_result = test_configuration(model_name, degraded_series, num_steps, guidance)
                    
                    # Calculate MAD if successful
                    if test_result['status'] == 'success':
                        metrics = calculate_mad(source_series, degraded_series, test_result['reconstructed'])
                    else:
                        metrics = {
                            'mad': np.nan,
                            'max_diff': np.nan,
                            'min_diff': np.nan,
                            'std_diff': np.nan,
                            'n_missing': 0
                        }
                    
                    # Store result
                    results.append({
                        'dataset': test_case['metadata']['dataset'],
                        'technique': test_case['metadata']['technique'],
                        'rate_percent': test_case['metadata']['rate_percent'],
                        'iteration': test_case['metadata']['iteration'],
                        'model': model_name,
                        'num_inference_steps': num_steps,
                        'guidance_scale': guidance,
                        'mad': metrics['mad'],
                        'max_diff': metrics['max_diff'],
                        'min_diff': metrics['min_diff'],
                        'std_diff': metrics['std_diff'],
                        'n_missing': metrics['n_missing'],
                        'time_seconds': test_result['time_seconds'],
                        'status': test_result['status'],
                        'error': test_result['error']
                    })
                    
                    pbar.update(1)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate efficiency metrics
    df_results['quality_score'] = 1 / (df_results['mad'] + 0.001)  # Lower MAD = higher quality
    df_results['time_per_step'] = df_results['time_seconds'] / df_results['num_inference_steps']
    df_results['quality_per_second'] = df_results['quality_score'] / df_results['time_seconds']
    
    # Save detailed results
    output_file = output_dir / f"optimization_results_global_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved detailed results to: {output_file}")
    
    # Generate summary report
    print("\n" + "="*70)
    print("📊 GLOBAL OPTIMIZATION RESULTS")
    print("="*70)
    
    # Filter successful results
    df_success = df_results[df_results['status'] == 'success'].copy()
    
    if df_success.empty:
        print("❌ No successful tests!")
        return
    
    print(f"\n✓ Successfully completed: {len(df_success)}/{len(df_results)} tests")
    print(f"✓ Failed: {len(df_results) - len(df_success)} tests")
    
    # Aggregate results by hyperparameters (average across all datasets)
    agg_results = df_success.groupby(['model', 'num_inference_steps', 'guidance_scale']).agg({
        'mad': 'mean',
        'time_seconds': 'mean',
        'quality_per_second': 'mean'
    }).reset_index()
    
    # Best configurations by different criteria
    print("\n🏆 GLOBALLY OPTIMAL CONFIGURATIONS:")
    print("\n1. Best Average MAD (Lowest error across all datasets):")
    best_mad = agg_results.nsmallest(5, 'mad')
    for idx, row in best_mad.iterrows():
        print(f"   {row['model']}: steps={int(row['num_inference_steps'])}, guidance={row['guidance_scale']:.1f}")
        print(f"      Avg MAD={row['mad']:.4f}, Avg Time={row['time_seconds']:.1f}s")
    
    print("\n2. Best Quality/Time (Most efficient globally):")
    best_efficiency = agg_results.nlargest(5, 'quality_per_second')
    for idx, row in best_efficiency.iterrows():
        print(f"   {row['model']}: steps={int(row['num_inference_steps'])}, guidance={row['guidance_scale']:.1f}")
        print(f"      Avg MAD={row['mad']:.4f}, Avg Time={row['time_seconds']:.1f}s, Efficiency={row['quality_per_second']:.4f}")
    
    # Recommendations per model
    print("\n💡 OPTIMAL HYPERPARAMETERS PER MODEL:")
    for model_name in SD_MODELS:
        if model_name in agg_results['model'].values:
            model_results = agg_results[agg_results['model'] == model_name]
            best = model_results.loc[model_results['quality_per_second'].idxmax()]
            print(f"\n   {model_name}:")
            print(f"      num_inference_steps: {int(best['num_inference_steps'])}")
            print(f"      guidance_scale: {best['guidance_scale']:.1f}")
            print(f"      Expected avg MAD: {best['mad']:.4f}")
            print(f"      Expected avg time: {best['time_seconds']:.1f}s/dataset")
    
    # Analysis by hyperparameter
    print("\n📈 HYPERPARAMETER EFFECT ANALYSIS:")
    print("\nEffect of num_inference_steps (averaged across all datasets, techniques, rates):")
    steps_analysis = df_success.groupby('num_inference_steps').agg({
        'mad': ['mean', 'std'],
        'time_seconds': 'mean'
    }).round(4)
    steps_analysis.columns = ['mad_mean', 'mad_std', 'time_mean']
    print(steps_analysis.to_string())
    
    print("\nEffect of guidance_scale (averaged across all datasets, techniques, rates):")
    guidance_analysis = df_success.groupby('guidance_scale').agg({
        'mad': ['mean', 'std'],
        'time_seconds': 'mean'
    }).round(4)
    guidance_analysis.columns = ['mad_mean', 'mad_std', 'time_mean']
    print(guidance_analysis.to_string())
    
    # Generate config recommendations
    print("\n" + "="*70)
    print("⚙️  RECOMMENDED CONFIG.YAML SETTINGS (GLOBALLY OPTIMAL):")
    print("="*70)
    
    # Find overall best balanced configuration
    overall_best = agg_results.loc[agg_results['quality_per_second'].idxmax()]
    
    print(f"""
computation:
  stable_diffusion:
    num_inference_steps: {int(overall_best['num_inference_steps'])}  # Globally optimized
    guidance_scale: {overall_best['guidance_scale']:.1f}
    device: "cuda"  # or "cpu"
    
# Expected performance (averaged across all datasets):
#   Average MAD: {overall_best['mad']:.4f}
#   Average time per reconstruction: ~{overall_best['time_seconds']:.1f}s
#   Quality/Time efficiency: {overall_best['quality_per_second']:.4f}
#
# Tested on:
#   - {len(test_cases)} degraded datasets
#   - {len(datasets)} unique source datasets
#   - {len(techniques)} missingness techniques
#   - {len(rates)} missing rates
""")
    
    print("="*70)
    print("✅ Global optimization complete!")
    print(f"📁 Detailed results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()

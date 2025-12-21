#!/usr/bin/env python3
"""
Dataset Reconstruction Script
Repairs degraded univariate time series datasets using various reconstruction models.
Uses config.yaml for configuration.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List
import sys

# Import reconstruction models and config loader
from reconstruction_models import RECONSTRUCTION_MODELS
from config_loader import load_config


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


def reconstruct_dataset(degraded_file: str,
                       output_file: str,
                       reconstruction_model: str,
                       config = None) -> None:
    """
    Reconstruct a single degraded dataset.
    
    Args:
        degraded_file: Path to degraded CSV file
        output_file: Path to output reconstructed CSV file
        reconstruction_model: Name of reconstruction model to use
        config: Configuration object (for SD model settings)
    """
    # Load degraded dataset
    df = load_degraded_dataset(degraded_file)
    series = df.iloc[:, 0]  # First column is the time series
    
    # Get reconstruction model function
    if reconstruction_model not in RECONSTRUCTION_MODELS:
        raise ValueError(f"Unknown reconstruction model: {reconstruction_model}")
    
    model_func = RECONSTRUCTION_MODELS[reconstruction_model]
    
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
    
    # Create output DataFrame with original timestamps
    output_df = df.copy()
    output_df.iloc[:, 0] = reconstructed_series.values
    
    # Save reconstructed dataset
    output_df.to_csv(output_file)
    print(f"    ✓ Saved to: {output_file}")


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
    
    # Track progress
    completed = 0
    skipped = 0
    
    # Process each combination
    for degraded_file in filtered_files:
        try:
            metadata = parse_degraded_filename(degraded_file.name)
            
            print(f"\n{'='*70}")
            print(f"Processing: {degraded_file.name}")
            print(f"  Dataset: {metadata['dataset']}")
            print(f"  Technique: {metadata['technique']}")
            print(f"  Rate: {metadata['rate_percent']}%")
            print(f"  Iteration: {metadata['iteration']}")
            print(f"{'='*70}")
            
            for model_name in models:
                # Generate output filename
                # Format: datasetName_technique_rateP_iteration_model.csv
                base_name = degraded_file.stem
                output_filename = f"{base_name}_{model_name}.csv"
                output_file = os.path.join(output_dir, output_filename)
                
                # Check if file already exists
                # CLI --force flag overrides config setting
                overwrite = args.force if args.force else config.get_overwrite_existing()
                if Path(output_file).exists() and not overwrite:
                    print(f"  ⏭️  Skipping {model_name} (already exists, use --force to overwrite)")
                    skipped += 1
                    completed += 1
                    continue
                
                # Reconstruct dataset
                try:
                    reconstruct_dataset(
                        degraded_file=str(degraded_file),
                        output_file=output_file,
                        reconstruction_model=model_name,
                        config=config
                    )
                    completed += 1
                except Exception as e:
                    print(f"  ❌ Error with {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Print progress
                progress = (completed / total_operations) * 100
                print(f"  Progress: {completed}/{total_operations} ({progress:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error processing {degraded_file.name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("RECONSTRUCTION COMPLETE")
    print("="*70)
    print(f"✓ Completed: {completed - skipped}/{total_operations}")
    print(f"⏭️  Skipped (existing): {skipped}")
    print(f"📁 Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()


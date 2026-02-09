#!/usr/bin/env python3
"""
Data Splitting Script

This script splits cleaned univariate time series datasets into training and test sets.
The split is based on the last N samples defined in config.yaml going to the test set,
with the remaining samples going to the training set.

This temporal split preserves the time series structure and is appropriate for:
- Reconstruction evaluation (training data)
- Future prediction evaluation (test data)

Usage:
    python 2_create_split.py [--input-dir DIR] [--output-dir DIR] [--test-samples N]

Examples:
    # Split all datasets from default directories using config/config.yaml settings
    python 2_create_split.py

    # Split with custom test samples count
    python 2_create_split.py --test-samples 100

    # Split specific dataset
    python 2_create_split.py --dataset vibration_sensor_S1.csv
"""

import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging

# Setup automatic logging to file
setup_logging("2_create_split")

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config


def split_time_series(df: pd.DataFrame, test_samples: int) -> tuple:
    """
    Split a time series DataFrame into training and test sets.
    
    The split is temporal: last N samples go to test, rest to train.
    This preserves the time series structure.
    
    Args:
        df: DataFrame with time series data (index + value columns)
        test_samples: Number of last samples to use for test set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    total_samples = len(df)
    
    if test_samples >= total_samples:
        raise ValueError(
            f"test_samples ({test_samples}) must be less than total samples ({total_samples})"
        )
    
    if test_samples <= 0:
        raise ValueError(f"test_samples must be positive, got {test_samples}")
    
    # Split: all but last N for training, last N for test
    train_df = df.iloc[:-test_samples].copy()
    test_df = df.iloc[-test_samples:].copy()
    
    return train_df, test_df


def split_dataset(input_file: str, 
                  train_output_file: str, 
                  test_output_file: str,
                  test_samples: int,
                  config) -> dict:
    """
    Split a single dataset into training and test sets.
    
    Args:
        input_file: Path to input CSV file (cleaned data)
        train_output_file: Path to output training CSV file
        test_output_file: Path to output test CSV file
        test_samples: Number of samples for test set
        config: Configuration object
        
    Returns:
        Dictionary with split statistics
    """
    print(f"\n📂 Splitting: {os.path.basename(input_file)}")
    
    # Check if output files already exist
    train_exists = os.path.exists(train_output_file)
    test_exists = os.path.exists(test_output_file)
    
    if train_exists and test_exists and not config.get_overwrite_existing():
        print(f"  ⏭️  Skipping (files already exist, overwrite_existing=false)")
        return {'status': 'skipped'}
    
    # Load cleaned dataset
    try:
        df = pd.read_csv(input_file, index_col=0)
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return {'status': 'error', 'message': str(e)}
    
    total_samples = len(df)
    print(f"  📊 Total samples: {total_samples}")
    
    # Validate test_samples for this dataset
    if test_samples >= total_samples:
        print(f"  ⚠️  Warning: test_samples ({test_samples}) >= total samples ({total_samples})")
        print(f"      Using {total_samples // 5} samples for test (20% of data)")
        actual_test_samples = max(1, total_samples // 5)
    else:
        actual_test_samples = test_samples
    
    # Perform split
    try:
        train_df, test_df = split_time_series(df, actual_test_samples)
    except Exception as e:
        print(f"  ❌ Error splitting: {e}")
        return {'status': 'error', 'message': str(e)}
    
    train_samples = len(train_df)
    test_samples_actual = len(test_df)
    
    print(f"  📈 Train samples: {train_samples} ({train_samples/total_samples*100:.1f}%)")
    print(f"  📉 Test samples:  {test_samples_actual} ({test_samples_actual/total_samples*100:.1f}%)")
    
    # Save split datasets
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)
    
    train_df.to_csv(train_output_file)
    test_df.to_csv(test_output_file)
    
    print(f"  ✅ Saved train: {train_output_file}")
    print(f"  ✅ Saved test:  {test_output_file}")
    
    return {
        'status': 'success',
        'total_samples': total_samples,
        'train_samples': train_samples,
        'test_samples': test_samples_actual
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Split cleaned time series datasets into training and test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config/config.yaml
  python 2_create_split.py
  
  # Override test samples count
  python 2_create_split.py --test-samples 100
  
  # Split specific dataset
  python 2_create_split.py --dataset vibration_sensor_S1.csv
        """
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing cleaned datasets (default: from config/config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for split datasets (default: data/2_splitted_data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specific dataset filename to split (default: all datasets)'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        help='Number of last samples for test set (default: from config/config.yaml)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    # Determine directories and parameters
    input_dir = args.input_dir or config.get_cleaned_dir()
    output_base_dir = args.output_dir or config.get_splitted_dir()
    train_output_dir = os.path.join(output_base_dir, 'train')
    test_output_dir = os.path.join(output_base_dir, 'test')
    test_samples = args.test_samples or config.get_test_samples()
    
    print(f"\n{'='*60}")
    print(f"📊 DATA SPLITTING PIPELINE")
    print(f"{'='*60}")
    print(f"Input directory:       {input_dir}")
    print(f"Train output directory: {train_output_dir}")
    print(f"Test output directory:  {test_output_dir}")
    print(f"Test samples (last N):  {test_samples}")
    
    # Get list of datasets to split
    if args.dataset:
        # Split specific dataset
        datasets = [args.dataset]
    else:
        # Split all CSV files in input directory
        if not os.path.exists(input_dir):
            print(f"\n❌ Error: Input directory does not exist: {input_dir}")
            return
        
        datasets = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not datasets:
        print(f"\n⚠️  No CSV files found in {input_dir}")
        return
    
    print(f"\n📋 Found {len(datasets)} dataset(s) to split")
    
    # Split each dataset
    success_count = 0
    skip_count = 0
    error_count = 0
    
    total_train_samples = 0
    total_test_samples = 0
    
    for dataset in datasets:
        input_file = os.path.join(input_dir, dataset)
        train_output_file = os.path.join(train_output_dir, dataset)
        test_output_file = os.path.join(test_output_dir, dataset)
        
        try:
            result = split_dataset(
                input_file, 
                train_output_file, 
                test_output_file,
                test_samples,
                config
            )
            
            if result['status'] == 'success':
                success_count += 1
                total_train_samples += result['train_samples']
                total_test_samples += result['test_samples']
            elif result['status'] == 'skipped':
                skip_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"\n❌ Error splitting {dataset}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ SPLITTING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully split: {success_count}/{len(datasets)} datasets")
    print(f"Skipped (existing): {skip_count}")
    print(f"Errors:             {error_count}")
    
    if success_count > 0:
        print(f"\n📊 Total samples split:")
        print(f"   Train: {total_train_samples}")
        print(f"   Test:  {total_test_samples}")
    
    print(f"\n📁 Output locations:")
    print(f"   Train: {train_output_dir}")
    print(f"   Test:  {test_output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

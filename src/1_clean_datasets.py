"""
Data Cleaning Script

This script loads raw datasets from data/0_source_data, cleans and validates them,
then saves the cleaned versions to data/1_cleaned_data.

Cleaning operations:
- Validates and converts index to datetime or numeric format
- Converts value columns to numeric (float)
- Handles different CSV formats (separator, decimal point)
- Removes duplicate indices
- Removes rows with invalid data
- Standardizes output format

Usage:
    python clean_datasets.py [--input-dir DIR] [--output-dir DIR] [--dataset FILENAME]

Examples:
    # Clean all datasets from default directories
    python clean_datasets.py

    # Clean specific dataset
    python clean_datasets.py --dataset vibration_sensor_S1.csv

    # Clean from custom directories
    python clean_datasets.py --input-dir data/0_source_data --output-dir data/1_cleaned_data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config


def detect_csv_format(file_path: str) -> dict:
    """
    Detect CSV format by reading first few lines.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary with pandas read_csv parameters
    """
    # Try common formats
    formats = [
        {'sep': ',', 'decimal': '.'},  # Standard format
        {'sep': ';', 'decimal': ','},  # European format
        {'sep': ';', 'decimal': '.'},  # Semicolon with dot
        {'sep': ',', 'decimal': ','},  # Comma with comma (rare)
    ]
    
    for fmt in formats:
        try:
            df = pd.read_csv(file_path, nrows=5, **fmt)
            if len(df.columns) >= 2 and len(df) > 0:
                # Check if we can convert at least one value to float
                for col in df.columns[1:]:  # Skip first column (index)
                    try:
                        pd.to_numeric(df[col].iloc[0])
                        return fmt
                    except (ValueError, TypeError):
                        continue
        except Exception:
            continue
    
    # Default to standard format
    return {'sep': ',', 'decimal': '.'}


def clean_index(index_series: pd.Series) -> pd.Index:
    """
    Clean and convert index to appropriate type.
    
    Args:
        index_series: Raw index series
        
    Returns:
        Cleaned pandas Index (datetime, numeric, or string)
    """
    # Try datetime first
    try:
        return pd.to_datetime(index_series)
    except (ValueError, TypeError, pd.errors.ParserError):
        pass
    
    # Try numeric
    try:
        numeric_index = pd.to_numeric(index_series, errors='coerce')
        if numeric_index.notna().all():
            return pd.Index(numeric_index)
    except (ValueError, TypeError):
        pass
    
    # Keep as string if both conversions fail
    return pd.Index(index_series.astype(str))


def clean_dataset(input_file: str, output_file: str, config) -> None:
    """
    Clean a single dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        config: Configuration object
    """
    print(f"\n📂 Cleaning: {os.path.basename(input_file)}")
    
    # Check if output file already exists
    if os.path.exists(output_file) and not config.get_overwrite_existing():
        print(f"  ⏭️  Skipping (file already exists, overwrite_existing=false)")
        return
    
    # Detect CSV format
    csv_format = detect_csv_format(input_file)
    print(f"  📋 Detected format: sep='{csv_format['sep']}', decimal='{csv_format['decimal']}'")
    
    # Load dataset
    try:
        df = pd.read_csv(input_file, **csv_format)
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return
    
    if len(df.columns) < 2:
        print(f"  ❌ Error: Dataset must have at least 2 columns (index + values)")
        return
    
    original_rows = len(df)
    print(f"  📊 Original rows: {original_rows}")
    
    # Set first column as index
    df.set_index(df.columns[0], inplace=True)
    
    # Clean index
    try:
        df.index = clean_index(df.index.to_series())
        print(f"  🔢 Index type: {df.index.dtype}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not convert index: {e}")
    
    # Convert value columns to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"  ⚠️  Warning: Could not convert column '{col}' to numeric: {e}")
    
    # Remove rows where all value columns are NaN
    df.dropna(how='all', inplace=True)
    
    # Remove duplicate indices (keep first occurrence)
    if df.index.duplicated().any():
        n_duplicates = df.index.duplicated().sum()
        print(f"  🔄 Removing {n_duplicates} duplicate indices")
        df = df[~df.index.duplicated(keep='first')]
    
    # Sort by index
    try:
        df.sort_index(inplace=True)
    except TypeError:
        # If index is not sortable, keep original order
        pass
    
    final_rows = len(df)
    print(f"  📊 Final rows: {final_rows} ({final_rows - original_rows:+d})")
    
    # Save cleaned dataset (standardized format: comma separator, dot decimal)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, sep=',', decimal='.')
    print(f"  ✅ Saved to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Clean raw datasets and save to cleaned data directory"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing raw datasets (default: from config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for cleaned datasets (default: data/1_cleaned_data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Specific dataset filename to clean (default: all datasets)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Determine directories
    # Input from 0_source_data (raw data)
    input_dir = args.input_dir or config.get_raw_source_dir()
    # Output to 1_cleaned_data
    output_dir = args.output_dir or config.get_cleaned_dir()
    
    print(f"\n{'='*60}")
    print(f"🧹 DATA CLEANING PIPELINE")
    print(f"{'='*60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get list of datasets to clean
    if args.dataset:
        # Clean specific dataset
        datasets = [args.dataset]
    else:
        # Clean all CSV files in input directory
        if not os.path.exists(input_dir):
            print(f"\n❌ Error: Input directory does not exist: {input_dir}")
            return
        
        datasets = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not datasets:
        print(f"\n⚠️  No CSV files found in {input_dir}")
        return
    
    print(f"\n📋 Found {len(datasets)} dataset(s) to clean")
    
    # Clean each dataset
    success_count = 0
    for dataset in datasets:
        input_file = os.path.join(input_dir, dataset)
        output_file = os.path.join(output_dir, dataset)
        
        try:
            clean_dataset(input_file, output_file, config)
            success_count += 1
        except Exception as e:
            print(f"\n❌ Error cleaning {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Cleaning complete: {success_count}/{len(datasets)} datasets cleaned successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


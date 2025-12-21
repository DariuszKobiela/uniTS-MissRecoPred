#!/usr/bin/env python3
"""
Example Usage of Time Series Reconstruction Framework
Demonstrates how to use the framework programmatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import modules
from missingness_techniques import MISSINGNESS_TECHNIQUES
from reconstruction_models import RECONSTRUCTION_MODELS


def example_basic_usage():
    """Basic example: degrade and reconstruct a single time series"""
    print("="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Create a simple time series
    t = np.linspace(0, 10, 100)
    data = pd.Series(np.sin(t) + 0.1 * np.random.randn(100))
    
    print(f"Original data: {len(data)} points")
    print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    
    # Apply MCAR missingness (5%)
    print("\n1. Applying MCAR missingness (5%)...")
    degraded = MISSINGNESS_TECHNIQUES['MCAR'](data, missing_rate=0.05, seed=42)
    print(f"  Missing values: {degraded.isna().sum()}/{len(degraded)}")
    
    # Reconstruct using linear interpolation
    print("\n2. Reconstructing with linear interpolation...")
    reconstructed = RECONSTRUCTION_MODELS['interpolate_linear'](degraded)
    print(f"  Reconstructed values: {len(reconstructed)}")
    
    # Calculate error
    print("\n3. Calculating reconstruction error...")
    error = np.abs(data - reconstructed).mean()
    print(f"  Mean Absolute Error: {error:.4f}")
    
    print("\n✓ Example complete!\n")


def example_compare_models():
    """Example: compare multiple reconstruction models"""
    print("="*70)
    print("EXAMPLE 2: Compare Multiple Models")
    print("="*70)
    
    # Create a simple time series
    t = np.linspace(0, 10, 100)
    data = pd.Series(np.sin(t) + 0.5 * np.cos(2*t) + 0.1 * np.random.randn(100))
    
    print(f"Original data: {len(data)} points")
    
    # Apply MAR missingness (10%)
    print("\n1. Applying MAR missingness (10%)...")
    degraded = MISSINGNESS_TECHNIQUES['MAR'](data, missing_rate=0.10, seed=42)
    print(f"  Missing values: {degraded.isna().sum()}/{len(degraded)}")
    
    # Test multiple models
    print("\n2. Testing multiple reconstruction models...")
    models_to_test = [
        'impute_mean',
        'interpolate_linear',
        'interpolate_cubic',
        'knn'
    ]
    
    results = []
    for model_name in models_to_test:
        print(f"\n  Testing {model_name}...")
        try:
            reconstructed = RECONSTRUCTION_MODELS[model_name](degraded.copy())
            mae = np.abs(data - reconstructed).mean()
            results.append({'model': model_name, 'mae': mae})
            print(f"    MAE: {mae:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Sort by MAE
    results.sort(key=lambda x: x['mae'])
    
    print("\n3. Results (sorted by MAE):")
    print("  " + "-"*40)
    for r in results:
        print(f"  {r['model']:25s} MAE: {r['mae']:.4f}")
    print("  " + "-"*40)
    
    print(f"\n✓ Best model: {results[0]['model']} (MAE: {results[0]['mae']:.4f})")
    print("\n✓ Example complete!\n")


def example_load_real_dataset():
    """Example: load and process real dataset"""
    print("="*70)
    print("EXAMPLE 3: Load Real Dataset")
    print("="*70)
    
    # Try to load a real dataset
    dataset_path = "data/0_source_data/boiler_outlet_temp_univ.csv"
    
    if not Path(dataset_path).exists():
        print(f"⚠️  Dataset not found: {dataset_path}")
        print("   This example requires source data to be present.")
        return
    
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, index_col=0)
    data = df.iloc[:, 0]
    
    print(f"  Total points: {len(data)}")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std: {data.std():.4f}")
    print(f"  Min: {data.min():.4f}")
    print(f"  Max: {data.max():.4f}")
    
    # Use first 500 points for demo
    data_subset = data.head(500).reset_index(drop=True)
    print(f"\nUsing first {len(data_subset)} points for demo...")
    
    # Apply MNAR missingness
    print("\n1. Applying MNAR missingness (5%)...")
    degraded = MISSINGNESS_TECHNIQUES['MNAR'](data_subset, missing_rate=0.05, seed=42)
    print(f"  Missing values: {degraded.isna().sum()}/{len(degraded)}")
    
    # Reconstruct
    print("\n2. Reconstructing with cubic interpolation...")
    reconstructed = RECONSTRUCTION_MODELS['interpolate_cubic'](degraded)
    
    # Calculate error
    print("\n3. Calculating error...")
    mae = np.abs(data_subset - reconstructed).mean()
    max_error = np.abs(data_subset - reconstructed).max()
    
    print(f"  MAE: {mae:.4f}")
    print(f"  Max Error: {max_error:.4f}")
    print(f"  Relative MAE: {mae / data_subset.std():.4f} (in terms of std devs)")
    
    print("\n✓ Example complete!\n")


def example_all_techniques():
    """Example: test all missingness techniques"""
    print("="*70)
    print("EXAMPLE 4: Compare All Missingness Techniques")
    print("="*70)
    
    # Create a simple time series
    t = np.linspace(0, 10, 200)
    data = pd.Series(5 + 2*np.sin(t) + np.cos(3*t) + 0.1 * np.random.randn(200))
    
    print(f"Original data: {len(data)} points")
    
    missing_rate = 0.10
    model_name = 'interpolate_linear'
    
    print(f"\nTesting all techniques with:")
    print(f"  Missing rate: {missing_rate*100}%")
    print(f"  Reconstruction: {model_name}")
    
    results = []
    
    for technique_name in MISSINGNESS_TECHNIQUES.keys():
        print(f"\n{technique_name}:")
        
        # Apply missingness
        degraded = MISSINGNESS_TECHNIQUES[technique_name](data, missing_rate, seed=42)
        print(f"  Missing: {degraded.isna().sum()} values")
        
        # Reconstruct
        reconstructed = RECONSTRUCTION_MODELS[model_name](degraded)
        
        # Calculate error
        mae = np.abs(data - reconstructed).mean()
        print(f"  MAE: {mae:.4f}")
        
        results.append({'technique': technique_name, 'mae': mae})
    
    # Sort results
    results.sort(key=lambda x: x['mae'])
    
    print("\n" + "="*70)
    print("Summary (sorted by MAE):")
    print("-"*70)
    for r in results:
        print(f"  {r['technique']:10s} MAE: {r['mae']:.4f}")
    print("="*70)
    
    print("\n✓ Example complete!\n")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "TIME SERIES RECONSTRUCTION EXAMPLES" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    try:
        example_basic_usage()
        input("Press Enter to continue to Example 2...")
        
        example_compare_models()
        input("Press Enter to continue to Example 3...")
        
        example_load_real_dataset()
        input("Press Enter to continue to Example 4...")
        
        example_all_techniques()
        
        print("="*70)
        print("ALL EXAMPLES COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run: python degrade_datasets.py --help")
        print("  2. Run: python reconstruct_datasets.py --help")
        print("  3. Run: python calculate_differences.py --help")
        print("  4. Run: streamlit run visualization.py")
        print("\nOr use Makefile/PowerShell script:")
        print("  make pipeline    (Linux/Mac)")
        print("  .\\run.ps1 pipeline  (Windows)")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


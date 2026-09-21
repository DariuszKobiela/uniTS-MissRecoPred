#!/usr/bin/env python3
"""
Dataset Degradation Script
Introduces missing values into training time series datasets.
Uses config/config.yaml for configuration.

NOTE: This script operates on TRAINING data only (from data/2_splitted_data/train/).
Test data is preserved separately for prediction evaluation.
"""

import os
import sys
import argparse
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logging

# Setup automatic logging to file
setup_logging("8_degrade_datasets")

from framework.plugin_registry import get_missingness_techniques
from missingness_techniques.structured import apply_structured_missingness
from utils.config_loader import load_config
from utils.experiment_naming import encode_missingness_label
from utils.missingness_analysis import summarize_missingness


def load_source_dataset(file_path: str, config) -> pd.DataFrame:
    """
    Load a source dataset with proper handling of different CSV formats.
    
    Args:
        file_path: Path to source CSV file
        config: Config object with format settings
        
    Returns:
        DataFrame with timestamps as index
    """
    # Get format settings for this file
    format_settings = config.get_csv_format(os.path.basename(file_path))
    
    df = pd.read_csv(
        file_path,
        **format_settings
    )
    
    return df


def process_single_degradation(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for one reproducible mechanism × structure realization."""
    metadata = {
        "dataset_name": Path(task["source_file"]).stem,
        "mechanism": task["technique"],
        "structure": task["structure"],
        "requested_missing_rate": task["rate"],
        "rate_percent": int(task["rate"] * 100),
        "iteration": task["iteration"],
        "seed": task["seed"],
        "output_file": task["output_file"],
    }
    try:
        if Path(task["output_file"]).exists() and not task["force"]:
            existing = pd.read_csv(task["output_file"], index_col=0)
            series = pd.to_numeric(existing.iloc[:, 0], errors="coerce")
            summary, gaps = summarize_missingness(series, metadata=metadata)
            return {
                "status": "skipped",
                "message": "Already exists",
                "output_file": task["output_file"],
                "summary": summary,
                "gaps": gaps,
            }

        summary, gaps = degrade_dataset(
            source_file=task["source_file"],
            output_file=task["output_file"],
            missingness_technique=task["technique"],
            missing_rate=task["rate"],
            seed=task["seed"],
            config=task["config"],
            structure=task["structure"],
            iteration=task["iteration"],
        )
        return {
            "status": "success",
            "message": "Completed",
            "output_file": task["output_file"],
            "summary": summary,
            "gaps": gaps,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "output_file": task["output_file"],
        }


def degrade_dataset(
    source_file: str,
    output_file: str,
    missingness_technique: str,
    missing_rate: float,
    seed: int | None = None,
    config=None,
    structure: str = "scattered",
    iteration: int | None = None,
) -> tuple[dict, list[dict]]:
    """Degrade one dataset and return realization- and gap-level diagnostics."""
    df = load_source_dataset(source_file, config)
    series = df.iloc[:, 0]
    all_techniques = get_missingness_techniques()
    if missingness_technique not in all_techniques:
        raise ValueError(f"Unknown missingness technique: {missingness_technique}")

    print(
        f"\n  Applying {missingness_technique} × {structure} "
        f"with rate {missing_rate*100:.1f}%..."
    )
    if missingness_technique in {"MCAR", "MAR", "MNAR"}:
        settings = config.get_missingness_structure_settings() if config else {}
        degraded_series = apply_structured_missingness(
            series,
            missing_rate,
            missingness_technique,
            structure,
            seed=seed,
            **settings,
        )
    elif structure == "scattered":
        degraded_series = all_techniques[missingness_technique](
            series, missing_rate, seed=seed
        )
    else:
        raise ValueError(
            f"Plugin mechanism {missingness_technique!r} supports only scattered "
            "missingness unless it implements structured missingness"
        )

    output_df = df.copy()
    output_df.iloc[:, 0] = degraded_series.values
    output_df.to_csv(output_file)
    print(f"  ✓ Saved to: {output_file}")
    metadata = {
        "dataset_name": Path(source_file).stem,
        "mechanism": missingness_technique,
        "structure": structure,
        "requested_missing_rate": missing_rate,
        "rate_percent": int(missing_rate * 100),
        "iteration": iteration,
        "seed": seed,
        "output_file": output_file,
    }
    return summarize_missingness(degraded_series, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Degrade univariate time series datasets by introducing missing values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from config/config.yaml
  python 8_degrade_datasets.py
  
  # Override config with custom parameters
  python 8_degrade_datasets.py --techniques MCAR --rates 0.05 --iterations 3
  
  # Use custom config file
  python 8_degrade_datasets.py --config config/my_config.yaml
  
  # Specify datasets by file paths (from training split)
  python 8_degrade_datasets.py --dataset-files data/2_splitted_data/train/boiler.csv
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--dataset-files',
        nargs='+',
        help='Specific dataset files to process (overrides config)'
    )
    
    parser.add_argument(
        '--techniques',
        nargs='+',
        help='Missingness techniques to apply (overrides config)'
    )
    
    parser.add_argument(
        '--rates',
        nargs='+',
        type=float,
        help='Missing rates as fractions (overrides config)'
    )

    parser.add_argument(
        '--structures',
        nargs='+',
        choices=['scattered', 'contiguous', 'mixed'],
        help='Temporal missingness structures (overrides config)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of iterations (overrides config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Base random seed (overrides config)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing degraded datasets'
    )
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        print("   Creating default config/config.yaml...")
        return

    run_degrade_datasets(
        config,
        dataset_files=args.dataset_files,
        techniques=args.techniques,
        rates=args.rates,
        structures=args.structures,
        iterations=args.iterations,
        seed=args.seed,
        force=args.force,
    )


def run_degrade_datasets(
    config,
    dataset_files: List[str] | None = None,
    techniques: List[str] | None = None,
    rates: List[float] | None = None,
    structures: List[str] | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    force: bool = False,
) -> bool:
    """Step 3: introduce missingness in training series."""
    if dataset_files:
        ds_files = dataset_files
    else:
        ds_files = config.get_datasets()

    if not ds_files:
        print("❌ No datasets found. Check your configuration or source directory.")
        return False

    techniques = techniques if techniques else config.get_missingness_techniques()
    rates = rates if rates else config.get_missingness_rates()
    structures = structures if structures else config.get_missingness_structures()
    iterations = iterations if iterations is not None else config.get_iterations()
    seed = seed if seed is not None else config.get_seed()
    output_dir = config.get_missing_dir()

    os.makedirs(output_dir, exist_ok=True)

    for rate in rates:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Invalid missing rate: {rate}. Must be between 0.0 and 1.0")

    total_operations = (
        len(ds_files) * len(techniques) * len(structures) * len(rates) * iterations
    )

    print("="*70)
    print("DATASET DEGRADATION")
    print("="*70)
    print(f"Datasets: {len(ds_files)} files")
    for ds in ds_files[:5]:
        print(f"  - {os.path.basename(ds)}")
    if len(ds_files) > 5:
        print(f"  ... and {len(ds_files) - 5} more")
    print(f"Techniques: {techniques}")
    print(f"Structures: {structures}")
    print(f"Missing rates: {[f'{r*100:.0f}%' for r in rates]}")
    print(f"Iterations: {iterations}")
    print(f"Base seed: {seed}")
    print(f"Total operations: {total_operations}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    print("\n📋 Building task list...")
    tasks = []
    for source_file in ds_files:
        if not Path(source_file).exists():
            print(f"❌ Source file not found: {source_file}")
            continue

        dataset_name = Path(source_file).stem

        for technique in techniques:
            for structure in structures:
                for rate in rates:
                    rate_percent = int(rate * 100)

                    for iteration in range(1, iterations + 1):
                        label = encode_missingness_label(technique, structure)
                        output_filename = (
                            f"{dataset_name}_{label}_{rate_percent}p_{iteration}.csv"
                        )
                        output_file = os.path.join(output_dir, output_filename)
                        seed_material = (
                            f"{seed}|{dataset_name}|{structure}|{rate_percent}|{iteration}"
                        ).encode("utf-8")
                        unique_seed = int.from_bytes(
                            hashlib.sha256(seed_material).digest()[:4], "big"
                        )

                        tasks.append({
                            'source_file': source_file,
                            'output_file': output_file,
                            'technique': technique,
                            'structure': structure,
                            'rate': rate,
                            'seed': unique_seed,
                            'iteration': iteration,
                            'config': config,
                            'force': force
                        })

    n_jobs = config.get_n_jobs()
    print(f"🚀 Processing {len(tasks)} tasks with {n_jobs} parallel job(s)...\n")

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_degradation)(task)
        for task in tqdm(tasks, desc="⏳ Degrading datasets", unit="task", ncols=80)
    )

    completed = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = [result["summary"] for result in results if result.get("summary")]
    gaps = [gap for result in results for gap in result.get("gaps", [])]
    pd.DataFrame(summaries).to_csv(
        report_dir / "missingness_realizations.csv", index=False
    )
    pd.DataFrame(gaps).to_csv(report_dir / "missingness_gaps.csv", index=False)

    if errors > 0:
        print("\n❌ Errors occurred:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {os.path.basename(r['output_file'])}: {r['message']}")

    print("\n" + "="*70)
    print("DEGRADATION COMPLETE")
    print("="*70)
    print(f"✅ Completed: {completed}/{len(tasks)}")
    print(f"⏭️  Skipped (existing): {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Missingness reports: {report_dir}")
    print("="*70)
    return True


if __name__ == "__main__":
    main()


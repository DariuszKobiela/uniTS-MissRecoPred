#!/usr/bin/env python3
"""
Stable Diffusion Hyperparameter Optimization Script

This script optimizes num_inference_steps and guidance_scale on available degraded 
datasets and Stable Diffusion models using Bayesian optimization (Optuna).

Optimization Method:
- Optuna: Bayesian optimization with TPE sampler (Tree-structured Parzen Estimator).
  This uses a CONTINUOUS search space to find the exact optimal values.

Usage:
    # Full optimization
    python optimize_sd_hyperparams.py --steps-min 10 --steps-max 100 --guidance-min 3.0 --guidance-max 15.0
    
    # Quick test
    python optimize_sd_hyperparams.py --n-trials 20 --max-files 5
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import gc
from contextlib import contextmanager

# Add parent directory (src) to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from reconstruction_models import RECONSTRUCTION_MODELS

# Strict dependencies for this script
try:
    import torch
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as e:
    print(f"❌ Error: Missing required dependency: {e}")
    print("   Please install torch and optuna: pip install torch optuna")
    sys.exit(1)


class TeeLogger:
    """Logger that writes to both file and console"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr during model execution"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def cleanup_memory():
    """Aggressive memory cleanup to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def clear_model_cache():
    """Clear Stable Diffusion model cache to free memory."""
    print("   🧹 Clearing model cache and forcing garbage collection...")
    try:
        from reconstruction_models import (
            stable_diffusion_2_gaf, 
            stable_diffusion_2_mtf, 
            stable_diffusion_2_rp, 
            stable_diffusion_2_spec
        )
        
        modules = [
            stable_diffusion_2_gaf, 
            stable_diffusion_2_mtf, 
            stable_diffusion_2_rp, 
            stable_diffusion_2_spec
        ]
        
        for module in modules:
            if hasattr(module, '_MODEL_CACHE'):
                keys = list(module._MODEL_CACHE.keys())
                for k in keys:
                    del module._MODEL_CACHE[k]
                module._MODEL_CACHE.clear()
        
        cleanup_memory()
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not clear model cache: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Optimize Stable Diffusion hyperparameters using Bayesian Optimization (Optuna)."
    )
    
    # Continuous Search Space Definitions
    parser.add_argument(
        '--steps-min',
        type=int,
        default=10,
        help='Minimum num_inference_steps (default: 10)'
    )
    
    parser.add_argument(
        '--steps-max',
        type=int,
        default=100,
        help='Maximum num_inference_steps (default: 75)'
    )
    
    parser.add_argument(
        '--guidance-min',
        type=float,
        default=3.0,
        help='Minimum guidance_scale (default: 3.0)'
    )
    
    parser.add_argument(
        '--guidance-max',
        type=float,
        default=15.0,
        help='Maximum guidance_scale (default: 12.0)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=30,
        help='Limit number of degraded files to test (for quick testing). Default: 30 random files.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='hyperparameter_optimization',
        help='Output directory for results (default: hyperparameter_optimization)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of Optuna trials per SD model (default: 100).'
    )
    
    return parser.parse_args()


def parse_degraded_filename(filename: str) -> dict:
    """Parse degraded filename to extract metadata."""
    name_without_ext = filename.replace('.csv', '')
    parts = name_without_ext.split('_')
    if len(parts) < 4: return None
    
    rate_idx = None
    for i, part in enumerate(parts):
        if part.endswith('p') and part[:-1].isdigit():
            rate_idx = i
            break
    
    if rate_idx is None or rate_idx < 1: return None
    
    return {
        'dataset': '_'.join(parts[:rate_idx - 1]),
        'technique': parts[rate_idx - 1],
        'rate_percent': int(parts[rate_idx].replace('p', '')),
        'iteration': int(parts[rate_idx + 1])
    }


def load_datasets(source_path: str, degraded_path: str, config):
    """Load and align source and degraded datasets."""
    degraded = pd.read_csv(degraded_path, index_col=0, na_values=['', ' '])
    degraded.iloc[:, 0] = pd.to_numeric(degraded.iloc[:, 0], errors='coerce')
    
    format_settings = config.get_csv_format(os.path.basename(source_path))
    source = pd.read_csv(source_path, **format_settings)
    if len(source.columns) >= 2:
        source.set_index(source.columns[0], inplace=True)
    source.iloc[:, 0] = pd.to_numeric(source.iloc[:, 0], errors='coerce')
    
    try:
        degraded.index = pd.to_datetime(degraded.index)
        source.index = pd.to_datetime(source.index)
    except:
        pass
        
    return source.iloc[:, 0], degraded.iloc[:, 0]


def calculate_mad(source: pd.Series, degraded: pd.Series, reconstructed: pd.Series) -> float:
    """Calculate Mean Absolute Difference for missing values only."""
    common_index = source.index.intersection(degraded.index).intersection(reconstructed.index)
    source_vals = source.loc[common_index]
    reconstructed_vals = reconstructed.loc[common_index]
    missing_mask = degraded.loc[common_index].isna()
    
    if not missing_mask.any(): return np.nan
    
    diff = (source_vals[missing_mask] - reconstructed_vals[missing_mask]).abs()
    return diff.mean()


def get_stable_diffusion_models() -> list:
    """Discover Stable Diffusion models from registry."""
    sd_models = [
        name for name in RECONSTRUCTION_MODELS.keys() 
        if name.startswith('stable_diffusion_')
    ]
    return sorted(sd_models)


def test_configuration(model_name: str, series: pd.Series, num_steps: int, guidance: float) -> dict:
    """Run a single reconstruction test with suppressed output."""
    model_func = RECONSTRUCTION_MODELS[model_name]
    start_time = time.time()
    try:
        with suppress_output():
            reconstructed = model_func(series, num_inference_steps=num_steps, guidance_scale=guidance)
        elapsed = time.time() - start_time
        return {'reconstructed': reconstructed, 'time': elapsed, 'status': 'success'}
    except Exception as e:
        elapsed = time.time() - start_time
        cleanup_memory()
        return {'reconstructed': None, 'time': elapsed, 'status': 'error', 'error': str(e)}


def run_optimization(args, test_cases, SD_MODELS, config):
    """Run Optuna optimization loop with CONTINUOUS search space."""
    print("\n" + "="*70)
    print("🎯 STARTING BAYESIAN OPTIMIZATION (OPTUNA)")
    print("   Method: Tree-structured Parzen Estimator (TPE)")
    print(f"   Trials per model: {args.n_trials}")
    print(f"   Search Space:")
    print(f"     - Steps: [{args.steps_min}, {args.steps_max}] (Integer)")
    print(f"     - Guidance: [{args.guidance_min}, {args.guidance_max}] (Float)")
    print("="*70)
    
    clear_model_cache()
    all_results = []
    best_params_per_model = {}
    
    for model_idx, model_name in enumerate(SD_MODELS, 1):
        print(f"\n📊 Optimizing Model {model_idx}/{len(SD_MODELS)}: {model_name}")
        print(f"{'-'*70}")
        
        # Pre-load model to show loading logs
        print(f"   🔄 Pre-loading {model_name} into memory (first run takes time)...")
        try:
            model_func = RECONSTRUCTION_MODELS[model_name]
            # Create minimal dummy data to trigger load
            dummy_series = pd.Series(np.random.rand(10))
            dummy_series.iloc[1] = np.nan
            # Run 1 step inference - this triggers get_model() and shows prints
            model_func(dummy_series, num_inference_steps=1)
            print("   ✓ Model loaded successfully.")
        except Exception as e:
            print(f"   ⚠️ Warning: Pre-loading check failed: {e}")

        def objective(trial):
            # Sample hyperparameters from CONTINUOUS distribution
            steps = trial.suggest_int('num_inference_steps', args.steps_min, args.steps_max)
            guidance = trial.suggest_float('guidance_scale', args.guidance_min, args.guidance_max, step=0.1)
            
            total_mad = 0
            n_success = 0
            
            for case in test_cases:
                try:
                    source, degraded = load_datasets(str(case['source_file']), str(case['degraded_file']), config)
                    result = test_configuration(model_name, degraded, steps, guidance)
                    
                    if result['status'] == 'success':
                        mad = calculate_mad(source, degraded, result['reconstructed'])
                        if not np.isnan(mad):
                            total_mad += mad
                            n_success += 1
                            
                            all_results.append({
                                'model': model_name,
                                'dataset': case['metadata']['dataset'],
                                'num_inference_steps': steps,
                                'guidance_scale': guidance,
                                'mad': mad,
                                'time': result['time'],
                                'status': 'success',
                                'trial': trial.number
                            })
                    
                    del source, degraded, result
                    cleanup_memory()
                    
                    if n_success > 0:
                        trial.report(total_mad / n_success, step=n_success)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                            
                except optuna.TrialPruned:
                    raise
                except Exception:
                    cleanup_memory()
                    continue
            
            if n_success == 0: return float('inf')
            return total_mad / n_success

        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        
        print(f"🚀 Running {args.n_trials} trials...")
        try:
            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
            
            best = study.best_params
            best_val = study.best_value
            best_params_per_model[model_name] = {
                'num_inference_steps': best['num_inference_steps'],
                'guidance_scale': best['guidance_scale'],
                'mad': best_val,
                'trials': len(study.trials)
            }
            print(f"✓ Best: steps={best['num_inference_steps']}, guidance={best['guidance_scale']:.1f}, MAD={best_val:.4f}")
            
        except Exception as e:
            print(f"❌ Optimization failed for {model_name}: {e}")
        
        clear_model_cache()
        
    return all_results, best_params_per_model


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"Summary_opt_res_{timestamp}.txt"
    logger = TeeLogger(summary_file)
    sys.stdout = logger
    
    try:
        SD_MODELS = get_stable_diffusion_models()
        if not SD_MODELS:
            print("❌ Error: No Stable Diffusion models found.")
            return

        print("="*70)
        print("🔬 STABLE DIFFUSION HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        
        config = load_config()
        missing_dir = Path(config.get_missing_dir())
        degraded_files = sorted(missing_dir.glob("*.csv"))
        
        if not degraded_files:
            print("❌ No degraded files found.")
            return
            
        if args.max_files:
            import random
            random.shuffle(degraded_files)
            degraded_files = degraded_files[:args.max_files]
            print(f"⚠️  Limiting to {args.max_files} random files.")
            
        test_cases = []
        source_dir = Path(config.get_source_dir())
        for f in degraded_files:
            meta = parse_degraded_filename(f.name)
            if meta:
                src = source_dir / f"{meta['dataset']}.csv"
                if src.exists():
                    test_cases.append({'degraded_file': f, 'source_file': src, 'metadata': meta})
        
        if not test_cases:
            print("❌ No valid test cases.")
            return
            
        print(f"✓ Prepared {len(test_cases)} test cases.")
        results, best_params = run_optimization(args, test_cases, SD_MODELS, config)
        
        if results:
            df = pd.DataFrame(results)
            csv_path = output_dir / f"optimization_results_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Saved detailed results to {csv_path}")
            
            print("\n" + "="*70)
            print("🏆 OPTIMIZATION SUMMARY")
            print("="*70)
            
            for model, params in best_params.items():
                print(f"\n{model}:")
                print(f"  Best Configuration: steps={params['num_inference_steps']}, guidance={params['guidance_scale']:.1f}")
                print(f"  Achieved MAD: {params['mad']:.4f}")

            if best_params:
                print("\n⚙️  RECOMMENDED CONFIG.YAML:")
                best_overall_model = min(best_params.items(), key=lambda x: x[1]['mad'])
                print(f"""
computation:
  stable_diffusion:
    num_inference_steps: {best_overall_model[1]['num_inference_steps']}
    guidance_scale: {best_overall_model[1]['guidance_scale']:.1f}
    device: "cuda"
""")
            
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n📄 Summary log saved to: {summary_file}")

if __name__ == "__main__":
    main()

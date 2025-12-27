# Time Series Reconstruction Framework

A modular framework for evaluating time series reconstruction methods on univariate datasets with various types of missing data.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Workflow](#-workflow)
- [Output Files](#-output-files)
- [Advanced Usage](#-advanced-usage)
- [Adding New Models](#-adding-new-models)
- [Visualization](#-visualization)
- [Citation](#-citation)

## ✨ Features

- **20+ Reconstruction Models**: From simple imputation to deep learning (Stable Diffusion 2)
- **3 Missingness Patterns**: MCAR, MAR, MNAR with configurable rates
- **Automatic Discovery**: Auto-detects datasets, models, and techniques
- **Configuration-Based**: YAML config for easy experiment management
- **Interactive Visualization**: Streamlit dashboard for result analysis
- **Modular Design**: Easy to add new models and techniques
- **MAD Metric**: Measures reconstruction quality only on missing values

## 📁 Project Structure

```
univariate-time-series-reconstruction-framework/
│
├── 📁 reconstruction_models/              # Reconstruction models (20 models)
│   ├── __init__.py                        # Registry of models
│   │
│   ├── impute_mean.py                     # Mean imputation
│   ├── impute_median.py                   # Median imputation
│   ├── impute_mode.py                     # Mode imputation
│   ├── impute_ffill.py                    # Forward fill
│   ├── impute_bfill.py                    # Backward fill
│   │
│   ├── interpolate_nearest.py             # Nearest neighbor interpolation
│   ├── interpolate_linear.py              # Linear interpolation
│   ├── interpolate_index.py               # Index-based interpolation
│   ├── interpolate_quadratic.py           # Quadratic interpolation
│   ├── interpolate_cubic.py               # Cubic interpolation
│   ├── interpolate_polynomial.py          # Polynomial interpolation
│   ├── interpolate_pchip.py               # PCHIP interpolation
│   ├── interpolate_akima.py               # Akima interpolation
│   ├── interpolate_spline.py              # Spline interpolation
│   │
│   ├── knn.py                             # K-Nearest Neighbors
│   ├── sarimax.py                         # SARIMA with Kalman smoothing
│   │
│   ├── stable_diffusion_2_gaf.py          # Stable Diffusion 2 + GAF
│   ├── stable_diffusion_2_mtf.py          # Stable Diffusion 2 + MTF
│   ├── stable_diffusion_2_rp.py           # Stable Diffusion 2 + RP
│   └── stable_diffusion_2_spec.py         # Stable Diffusion 2 + Spectrogram
│
├── 📁 missingness_techniques/             # Missingness patterns
│   ├── __init__.py                        # Registry of techniques
│   ├── mcar.py                            # Missing Completely At Random
│   ├── mar.py                             # Missing At Random
│   └── mnar.py                            # Missing Not At Random
│
├── 📁 data/
│   ├── 📁 0_source_data/                  # Original datasets (auto-discovered)
│   ├── 📁 2_missing_data/                 # Degraded datasets (generated)
│   └── 📁 3_fixed_data/                   # Reconstructed datasets (generated)
│
├── 📁 experiments_results/                # Results with timestamps
│
├── 🐍 1_clean_datasets.py                 # [MAIN] Clean and validate raw data
├── 🐍 2_degrade_datasets.py               # [MAIN] Introduce missing data
├── 🐍 3_reconstruct_datasets.py           # [MAIN] Reconstruct missing data
├── 🐍 4_calculate_mad.py                  # [MAIN] Calculate MAD metric
├── 🐍 5_visualize_mad_comparison.py       # [MAIN] Streamlit dashboard
│
├── 🐍 config_loader.py                    # Configuration manager
├── 🐍 example_usage.py                    # Usage examples
│
├── ⚙️ config.yaml                         # Main configuration file
├── ⚙️ Makefile                            # Automation (Linux/Mac)
│
├── 📝 requirements.txt                    # Python dependencies
├── 📝 .gitignore                          # Git ignore rules
└── 📝 README.md                           # This file
```

## 🚀 Quick Start

### 1. Setup Virtual Environment (Recommended)

Using a virtual environment ensures isolated dependencies and prevents conflicts.

#### Quick Setup (Recommended):

**Linux/Mac:**
```bash
# Run setup script (creates venv and installs dependencies)
chmod +x setup_venv.sh
./setup_venv.sh
```

**Windows (PowerShell):**
```powershell
# Run setup script (creates venv and installs dependencies)
.\setup_venv.ps1

# If you get execution policy error, run first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Manual Setup:

**Linux/Mac:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

**Windows (PowerShell):**

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# You should see (venv) in your terminal prompt
```

**Windows (CMD):**

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

#### Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```

### 2. Installation

**After activating your virtual environment**, install dependencies:

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

**New dependencies** (for performance monitoring):
- `psutil` - CPU and RAM monitoring
- `GPUtil` - GPU monitoring (optional, only if CUDA available)

If GPU monitoring is not needed, GPUtil can be skipped (performance metrics will still work without it).

### 3. Configure Experiment

Edit `config.yaml` to set:
- Datasets to use (auto-discovers all CSVs in `data/0_source_data/`)
- Reconstruction models to test
- Missingness techniques (MCAR, MAR, MNAR)
- Missing rates (e.g., 10%, 20%)
- Number of iterations

### 4. Run Pipeline

```bash
# Option 1: Run complete pipeline
make pipeline

# Option 2: Run step by step
python 1_clean_datasets.py        # Clean and validate raw data
python 2_degrade_datasets.py      # Create degraded datasets

# OPTIONAL: Optimize Stable Diffusion hyperparameters (run once)
python optimize_sd_hyperparams.py  # Find optimal num_inference_steps and guidance_scale
# Then update config.yaml with recommended values

python 3_reconstruct_datasets.py   # Reconstruct missing values
python 4_calculate_mad.py          # Calculate MAD metric
streamlit run 5_visualize_mad_comparison.py  # Visualize results
```

## ⚙️ Configuration

The framework uses `config.yaml` for all settings:

```yaml
# Data directories
data:
  raw_source_dir: "data/0_source_data"     # Raw input data
  cleaned_dir: "data/1_cleaned_data"       # Cleaned data
  source_dir: "data/1_cleaned_data"        # Source for degradation
  missing_dir: "data/2_missing_data"
  fixed_dir: "data/3_fixed_data"
  results_dir: "experiments_results"

# Datasets (empty = auto-discover all CSVs)
datasets:
  selected: []  # Or specify: ["dataset1.csv", "dataset2.csv"]
  
# Models (empty = use all available)
reconstruction_models:
  selected: []  # Or specify: ["interpolate_linear", "knn"]
  excluded:     # Exclude specific models (e.g., GPU-only)
    # - "stable_diffusion_2_gaf"
    
# Missingness techniques (empty = use all)
missingness_techniques:
  selected: []  # Or specify: ["MCAR", "MAR"]
  
# Missingness rates
missingness_rates:
  rates: [0.10, 0.20]  # 10%, 20%
  iterations: 2
  seed: 42

# Computation settings
computation:
  overwrite_existing: false  # Set to true to overwrite existing files
  n_jobs: 4                  # Parallel processing: 1=sequential, 4=use 4 cores, -1=use all cores
  
  # Stable Diffusion hyperparameters (use optimize_sd_hyperparams.py to find optimal values)
  stable_diffusion:
    num_inference_steps: 50  # Default: 50. Lower=faster but less accurate. Optimize with optimize_sd_hyperparams.py
    guidance_scale: 7.5      # Default: 7.5. Controls adherence to prompt (1-20)
    device: "cuda"           # "cuda" for GPU or "cpu"
```

### Parallel Processing

The framework supports **parallel processing** for faster experiments:

- **`n_jobs: 4`** - Use 4 CPU cores (3-4x faster)
- **`n_jobs: -1`** - Use all available CPU cores  
- **`n_jobs: 1`** - Sequential processing (disable parallelization)

**Smart GPU Detection**: GPU models (Stable Diffusion) automatically run sequentially even when `n_jobs > 1`, preventing GPU memory conflicts.

**Example speedup** (4 cores):
- 72 degradations: 10 min → 3 min
- 1,152 reconstructions: 40 min → 10 min

**Note**: Hyperparameter optimization (`optimize_sd_hyperparams.py`) runs sequentially on GPU models and may take several hours depending on the number of test cases and parameter combinations

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECONSTRUCTION WORKFLOW                       │
└─────────────────────────────────────────────────────────────────┘

0️⃣ DATA CLEANING (1_clean_datasets.py)
   ├─ Load: data/0_source_data/*.csv (raw data)
   ├─ Auto-detect CSV format (separator, decimal point)
   ├─ Validate & convert: index (datetime/numeric), values (float)
   ├─ Remove: duplicates, invalid rows
   └─ Save: data/1_cleaned_data/*.csv (standardized format)
   
   📊 Example: 6 raw datasets → 6 cleaned datasets

                              ↓

1️⃣ DEGRADATION (2_degrade_datasets.py)
   ├─ Auto-discover: data/1_cleaned_data/*.csv
   ├─ Apply: MCAR / MAR / MNAR
   ├─ Rates: 10%, 20% (from config.yaml)
   ├─ Iterations: 2x (from config.yaml)
   └─ Save: data/2_missing_data/{dataset}_{technique}_{rate}p_{iter}.csv
   
   📊 Example: 6 datasets × 3 techniques × 2 rates × 2 iterations = 72 files

                              ↓

2️⃣ RECONSTRUCTION (3_reconstruct_datasets.py)
   ├─ Load: data/2_missing_data/*.csv
   ├─ Apply: 20 reconstruction models (from config.yaml)
   ├─ Monitor: Time, CPU, RAM, GPU usage per reconstruction
   ├─ Save: data/3_fixed_data/{dataset}_{technique}_{rate}p_{iter}_{model}.csv
   └─ Save metrics: experiments_results/performance_metrics_YYYYMMDD_HHMMSS.csv
   
   📊 Example: 72 × 20 = 1,440 files + performance metrics

                              ↓

3️⃣ EVALUATION (4_calculate_mad.py)
   ├─ Load: data/3_fixed_data/*.csv
   ├─ Compare ONLY missing values with: data/1_cleaned_data/*.csv
   ├─ Calculate: MAD (Mean Absolute Difference)
   └─ Save: experiments_results/reconstruction_results_YYYYMMDD_HHMMSS.csv
   
   📊 Output: Single CSV with 1,440 rows
   
   ⚠️  IMPORTANT: MAD is calculated ONLY for values that were missing!

                              ↓

4️⃣ VISUALIZATION (5_visualize_mad_comparison.py)
   ├─ Load: experiments_results/*.csv (MAD results + performance metrics)
   ├─ Interactive Streamlit dashboard with 9 tabs:
   │  ├─ 📊 By Model: Compare reconstruction quality across models
   │  ├─ 🎯 By Technique: Compare MCAR vs MAR vs MNAR
   │  ├─ 📉 By Missing Rate: Compare 10% vs 20% etc.
   │  ├─ 📁 By Dataset: Compare performance per dataset
   │  ├─ 🔥 Heatmap: Overall performance matrix
   │  ├─ 🏆 Best/Worst: Top and bottom performing models
   │  ├─ ⏱️ Computation Time: Execution time analysis & complexity
   │  ├─ 💻 Resource Usage: CPU, RAM, GPU usage analysis
   │  └─ 📋 Raw Data: Searchable table with download
   └─ Export: Filtered results, plots
```

## 📊 Output Files

### Degraded Datasets
**Format**: `{dataset}_{technique}_{rate}p_{iteration}.csv`

**Example**: `boiler_MCAR_10p_1.csv`
```
└─┬──┘ └┬─┘  └─┬┘ └┘
  │     │      │   └─ Iteration: 1
  │     │      └───── Rate: 10%
  │     └─────────── Technique: MCAR
  └───────────────── Dataset: boiler
```

### Reconstructed Datasets
**Format**: `{dataset}_{technique}_{rate}p_{iteration}_{model}.csv`

**Example**: `boiler_MCAR_10p_1_interpolate_linear.csv`
```
└─┬──┘ └┬─┘  └─┬┘ └┘ └────────┬──────────┘
  │     │      │   │           └─ Model: interpolate_linear
  │     │      │   └───────────── Iteration: 1
  │     │      └───────────────── Rate: 10%
  │     └─────────────────────── Technique: MCAR
  └───────────────────────────── Dataset: boiler
```

### Results Files
**Format**: `reconstruction_results_YYYYMMDD_HHMMSS.csv`

**Columns**:
- `dataset_name` - Dataset name
- `technique` - Missingness technique (MCAR/MAR/MNAR)
- `rate_percent` - Missing rate (%)
- `iteration` - Iteration number
- `model` - Reconstruction model name
- `mad` - **Mean Absolute Difference** (only for missing values!)
- `max_diff` - Maximum difference
- `min_diff` - Minimum difference
- `std_diff` - Standard deviation of differences
- `n_missing` - Number of missing values reconstructed
- `n_total` - Total number of values in dataset

### Performance Metrics Files
**Format**: `performance_metrics_YYYYMMDD_HHMMSS.csv`

**Purpose**: Track computational complexity and resource usage for each reconstruction

**Columns**:
- `dataset` - Dataset name
- `technique` - Missingness technique
- `rate_percent` - Missing rate (%)
- `iteration` - Iteration number
- `model` - Reconstruction model name
- `time_seconds` - **Execution time in seconds**
- `cpu_percent` - **Average CPU usage (%)**
- `memory_mb` - **Peak RAM usage (MB)**
- `gpu_percent` - GPU utilization (%) - *null if GPU not available*
- `gpu_memory_mb` - GPU memory usage (MB) - *null if GPU not available*
- `timestamp` - When reconstruction was performed

**Use Cases**:
- Compare computational efficiency of models
- Identify resource-intensive algorithms
- Optimize for time vs. quality trade-offs
- Plan hardware requirements for large-scale experiments

## 🎯 Advanced Usage

### Custom Parameters via CLI

```bash
# Degrade specific datasets
python 2_degrade_datasets.py --dataset-files data/0_source_data/boiler.csv

# Use custom config
python 2_degrade_datasets.py --config my_config.yaml

# Override config parameters
python 2_degrade_datasets.py --techniques MCAR --rates 0.05 0.10 --iterations 3

# Reconstruct with specific models
python 3_reconstruct_datasets.py --models interpolate_linear knn

# Calculate with custom config
python 4_calculate_mad.py --config my_config.yaml
```

### Optimizing Stable Diffusion Hyperparameters

The framework includes a dedicated script to find **globally optimal** `num_inference_steps` and `guidance_scale` by testing on **ALL available degraded datasets**:

```bash
# Full optimization (tests on ALL degraded datasets, all 4 SD models)
python optimize_sd_hyperparams.py

# Custom parameter ranges
python optimize_sd_hyperparams.py \
  --steps 5 10 20 30 50 \
  --guidance 3.5 5.0 7.5 10.0

# Quick test (limit to first 5 files)
python optimize_sd_hyperparams.py \
  --steps 10 20 \
  --guidance 5.0 7.5 \
  --max-files 5
```

**How it works:**
- Automatically finds ALL degraded datasets in `data/2_missing_data/`
- Dynamically discovers ALL Stable Diffusion models from `reconstruction_models/` (any model starting with `stable_diffusion_*`)
- Tests every hyperparameter combination on every dataset and every SD model
- Aggregates results across all datasets, techniques, rates, iterations, and models
- Provides globally optimal recommendations for each model and overall best
- **Extensible**: Adding new SD models to `reconstruction_models/` automatically includes them in optimization

**Output:**
- Detailed CSV with all tested configurations per dataset
- Global aggregated statistics (average MAD, average time)
- Best configurations by:
  - Lowest average MAD (best quality globally)
  - Highest quality/time ratio (most efficient globally)
- Optimal hyperparameters per model
- Hyperparameter effect analysis with standard deviations
- Ready-to-use `config.yaml` recommendations

**Example output:**
```
🔬 STABLE DIFFUSION HYPERPARAMETER OPTIMIZATION
Testing on ALL available degraded datasets for global optimization
Models (4): stable_diffusion_2_gaf, stable_diffusion_2_mtf, stable_diffusion_2_rp, stable_diffusion_2_spec

📂 Found 48 degraded files to test
✓ Prepared 48 test cases

📊 Test case distribution:
   Datasets: 3 unique (vibration_sensor_S1, boiler, power_consumption)
   Techniques: MCAR=24, MAR=24
   Rates: 10%=24, 20%=24

🚀 Running 3840 total tests...

🏆 GLOBALLY OPTIMAL CONFIGURATIONS:

1. Best Average MAD (Lowest error across all datasets):
   stable_diffusion_2_gaf: steps=50, guidance=7.5
      Avg MAD=4.8234, Avg Time=92.3s
   stable_diffusion_2_rp: steps=30, guidance=7.5
      Avg MAD=5.1245, Avg Time=55.1s

2. Best Quality/Time (Most efficient globally):
   stable_diffusion_2_gaf: steps=20, guidance=7.5
      Avg MAD=5.0123, Avg Time=36.8s, Efficiency=0.0534

💡 OPTIMAL HYPERPARAMETERS PER MODEL:
   stable_diffusion_2_gaf:
      num_inference_steps: 20
      guidance_scale: 7.5
      Expected avg MAD: 5.0123
      Expected avg time: 36.8s/dataset
   
   stable_diffusion_2_mtf:
      num_inference_steps: 30
      guidance_scale: 7.5
      Expected avg MAD: 5.2456
      Expected avg time: 54.2s/dataset
   
   stable_diffusion_2_rp:
      num_inference_steps: 20
      guidance_scale: 5.0
      Expected avg MAD: 5.1789
      Expected avg time: 35.9s/dataset
   
   stable_diffusion_2_spec:
      num_inference_steps: 20
      guidance_scale: 7.5
      Expected avg MAD: 5.3012
      Expected avg time: 37.1s/dataset

⚙️  RECOMMENDED CONFIG.YAML SETTINGS (GLOBALLY OPTIMAL):

computation:
  stable_diffusion:
    num_inference_steps: 20  # Globally optimized
    guidance_scale: 7.5
    
# Expected performance (averaged across all datasets):
#   Average MAD: 5.0123
#   Average time per reconstruction: ~36.8s
#
# Tested on:
#   - 48 degraded datasets
#   - 3 unique source datasets
#   - 2 missingness techniques
#   - 2 missing rates
```

**⚠️ Important:** 
- Run optimization AFTER generating degraded datasets with `2_degrade_datasets.py`
- The script automatically discovers and tests ALL Stable Diffusion models from `reconstruction_models/`
- Models are detected by naming convention: `stable_diffusion_*` (e.g., `stable_diffusion_2_gaf`)
- Adding new SD models to the framework automatically includes them in optimization - no code changes needed!
- Optimization may take several hours depending on hardware, number of test cases, and number of SD models

### Available Models

#### Simple Imputation (5 models)
- `impute_mean` - Replace with mean
- `impute_median` - Replace with median
- `impute_mode` - Replace with mode
- `impute_ffill` - Forward fill
- `impute_bfill` - Backward fill

#### Interpolation (9 models)
- `interpolate_linear` - Linear interpolation
- `interpolate_cubic` - Cubic interpolation
- `interpolate_quadratic` - Quadratic interpolation
- `interpolate_nearest` - Nearest neighbor
- `interpolate_index` - Index-based
- `interpolate_polynomial` - Polynomial (order 2)
- `interpolate_pchip` - PCHIP (monotonic)
- `interpolate_akima` - Akima (smooth curves)
- `interpolate_spline` - Spline (order 2)

#### Advanced (6 models)
- `knn` - K-Nearest Neighbors
- `sarimax` - SARIMA with Kalman smoothing
- `stable_diffusion_2_gaf` - Stable Diffusion 2 + GAF encoding
- `stable_diffusion_2_mtf` - Stable Diffusion 2 + MTF encoding
- `stable_diffusion_2_rp` - Stable Diffusion 2 + RP encoding
- `stable_diffusion_2_spec` - Stable Diffusion 2 + Spectrogram

### Missingness Techniques

- **MCAR** (Missing Completely At Random): Random uniform distribution
- **MAR** (Missing At Random): Probability depends on deviation from median
- **MNAR** (Missing Not At Random): Probability increases over time (sensor degradation)

## 📈 Visualization Features

The Streamlit dashboard (`5_visualize_mad_comparison.py`) provides comprehensive analysis across **9 interactive tabs**:

### 1. 📊 By Model
Compare reconstruction quality (MAD) across all models with:
- Bar charts showing average MAD per model
- Model ranking visualization
- Statistical comparisons

### 2. 🎯 By Technique
Analyze performance differences between missingness techniques (MCAR vs MAR vs MNAR):
- Technique comparison charts
- Average MAD by technique
- Distribution analysis

### 3. 📉 By Missing Rate
Evaluate how reconstruction quality changes with missing data percentage:
- Line plots showing MAD vs missing rate
- Performance degradation curves
- Rate-specific analysis

### 4. 📁 By Dataset
Dataset-specific performance analysis:
- Per-dataset model comparisons
- Dataset difficulty assessment
- Cross-dataset patterns

### 5. 🔥 Heatmap
Matrix visualization showing:
- Model × Technique performance
- Model × Dataset performance
- Sortable by technique or dataset

### 6. 🏆 Best/Worst
Quick overview of top and bottom performers:
- Best 5 models (lowest MAD)
- Worst 5 models (highest MAD)
- Model comparison bar charts

### 7. ⏱️ Computation Time *(NEW)*
Analyze execution time and computational complexity:
- **Time summary**: Total, average, min, max execution time
- **Time by model**: Average execution time with standard deviation
- **Time distribution**: Box plots showing time variability
- **Time by technique/rate**: How missingness affects computation time
- **Detailed statistics**: Full breakdown per model

**Use cases**:
- Identify fast models for real-time applications
- Compare time-quality trade-offs
- Plan computational budgets

### 8. 💻 Resource Usage *(NEW)*
Monitor hardware resource consumption:
- **CPU usage**: Average and peak CPU utilization per model
- **RAM usage**: Memory consumption per model
- **GPU usage**: GPU utilization for deep learning models (if available)
- **Efficiency score**: Combined metric (time + CPU + RAM)
- **Time vs Memory scatter**: Visual efficiency comparison

**Metrics tracked**:
- CPU usage percentage
- RAM usage in MB
- GPU utilization percentage (optional)
- GPU memory usage in MB (optional)

**Use cases**:
- Identify resource-intensive models
- Optimize for limited hardware
- Plan cloud computing costs
- Balance accuracy vs. efficiency

### 9. 📋 Raw Data
Direct access to results:
- Searchable table with all metrics
- Sort by any column
- Download filtered results as CSV
- Full data transparency

### Global vs Local Filters
- **Global filters** (sidebar): Apply to ALL tabs
- **Local filters** (within tabs): Tab-specific refinement

## 📝 Adding New Models

### Add a Reconstruction Model

1. **Create** `reconstruction_models/my_model.py`:

```python
import pandas as pd

def my_model(data: pd.Series) -> pd.Series:
    """
    Your reconstruction logic.
    
    Args:
        data: Series with NaN values
        
    Returns:
        Series with reconstructed values
    """
    # Your implementation here
    return data.fillna(data.mean())  # Example
```

2. **Register** in `reconstruction_models/__init__.py`:

```python
from .my_model import my_model

RECONSTRUCTION_MODELS = {
    # ... existing models ...
    'my_model': my_model
}
```

3. **Use it**:

```bash
python 3_reconstruct_datasets.py --models my_model
```

### Add a Missingness Technique

1. **Create** `missingness_techniques/my_technique.py`:

```python
import pandas as pd
import numpy as np

def apply_my_technique(data: pd.Series, missing_rate: float, seed: int = None) -> pd.Series:
    """
    Your missingness logic.
    
    Args:
        data: Original series
        missing_rate: Fraction to make missing (0.0 to 1.0)
        seed: Random seed
        
    Returns:
        Series with NaN values
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_copy = data.copy()
    n_missing = int(len(data) * missing_rate)
    missing_indices = np.random.choice(len(data), n_missing, replace=False)
    data_copy.iloc[missing_indices] = np.nan
    
    return data_copy
```

2. **Register** in `missingness_techniques/__init__.py`:

```python
from .my_technique import apply_my_technique

MISSINGNESS_TECHNIQUES = {
    # ... existing techniques ...
    'MY_TECHNIQUE': apply_my_technique
}
```

3. **Use it**:

```bash
python 2_degrade_datasets.py --techniques MY_TECHNIQUE
```

## 🎨 Visualization

The Streamlit dashboard (`5_visualize_mad_comparison.py`) provides:

### Features
- **File Selection**: Choose results file from `experiments_results/`
- **Interactive Filters**: Dataset, Model, Technique, Missing Rate
- **Multiple Views**:
  - 📊 By Model - Compare reconstruction models
  - 🎯 By Technique - Compare missingness patterns
  - 📉 By Missing Rate - Analyze rate impact
  - 🔥 Heatmap - Model vs Technique matrix
  - 🏆 Best/Worst - Top and bottom performers
  - 📁 Raw Data - Filterable table with download

### Launch Dashboard

```bash
streamlit run 5_visualize_mad_comparison.py
# Open browser at http://localhost:8501
```

## 📚 Additional Documentation

- **[PERFORMANCE_METRICS.md](PERFORMANCE_METRICS.md)** - Detailed guide on computational performance tracking, resource monitoring, and efficiency analysis
- **[PARALLEL_PROCESSING.md](PARALLEL_PROCESSING.md)** - Guide on parallel processing implementation and performance optimization

## 🧹 Cleanup

### Clean Generated Files

```bash
# Remove generated files (keep source data)
make clean

# Remove everything including results
make clean-all
```

### Remove Virtual Environment

If you want to completely remove the virtual environment:

```bash
# First, deactivate if active
deactivate

# Remove venv directory
# Linux/Mac:
rm -rf venv

# Windows (PowerShell):
Remove-Item -Recurse -Force venv

# Windows (CMD):
rmdir /s /q venv
```

To recreate, just follow the setup steps again.

## 📈 Performance Estimates

### Timing (for 6 datasets × 3 techniques × 2 rates × 2 iterations = 72 degraded files)

**Sequential (`n_jobs: 1`)**:
- **Degradation**: ~5-10 minutes
- **Reconstruction (20 models)**: ~2-4 hours
- **Calculation**: ~10-20 minutes
- **Total Pipeline**: ~3-5 hours

**Parallel (`n_jobs: 4`)**:
- **Degradation**: ~2-3 minutes ⚡ **(3-4x faster)**
- **Reconstruction (16 CPU models)**: ~30-60 minutes ⚡ **(3-4x faster)**
- **Reconstruction (4 GPU models)**: ~30-90 minutes (sequential)
- **Calculation**: ~10-20 minutes
- **Total Pipeline**: ~1.5-2.5 hours ⚡ **(~2x faster overall)**

### Disk Space
- **Degraded data**: ~100-500 MB
- **Reconstructed data**: ~2-8 GB
- **Results**: ~1-5 MB per experiment

### Performance Tips
1. **For CPU models only**: Set `n_jobs: -1` (use all cores) and exclude SD models
2. **For mixed workloads**: Use `n_jobs: 4` with smart GPU detection (automatic)
3. **Large datasets**: Monitor RAM usage; reduce `n_jobs` if memory issues occur

## ⚠️ Important Notes

### Virtual Environment
Always activate your virtual environment before running any scripts:

```bash
# Linux/Mac
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat
```

You should see `(venv)` in your terminal prompt when active.

### MAD Metric
**MAD (Mean Absolute Difference)** measures reconstruction quality **ONLY for missing values**, not the entire series. This is crucial because:
- Non-missing values should remain unchanged
- We only care about how well the model reconstructed destroyed values
- Lower MAD = better reconstruction

### Stable Diffusion Models
- Require NVIDIA GPU with 8+ GB VRAM
- First run downloads ~20GB model from HuggingFace
- Cached for subsequent runs
- Can be excluded in `config.yaml` if no GPU available

**Safety Checker**: The framework disables Stable Diffusion's safety checker (`safety_checker=None`) because:
- We're processing **technical time series data**, not generating public images
- GAF/MTF/RP/Spectrograms are **mathematical visualizations**, not real-world images
- This is a **scientific research tool**, not a public service
- Disabling it significantly **speeds up** reconstruction
- The warning is automatically suppressed in the code (this is safe and expected)

## 🤝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{ts_reconstruction_framework_2025,
  title={Time Series Reconstruction Framework},
  author={Dariusz Kobiela, Jarosław Kobiela, Adam Kurowski, Agnieszka Landowska},
  year={2025},
  howpublished={GitHub repository},
  note={Framework for evaluating univariate time series reconstruction methods}
}
```

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Stable Diffusion 2 Model**: https://huggingface.co/Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec
- **Training Dataset**: https://huggingface.co/datasets/Daro77/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data

## 📧 Support

For issues, questions, or contributions:
1. Check existing documentation
2. Review example usage: `python example_usage.py`
3. Open an issue on GitHub

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Production Ready ✅

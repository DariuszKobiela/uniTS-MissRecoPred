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
├── 🐍 degrade_datasets.py                 # [MAIN] Introduce missing data
├── 🐍 reconstruct_datasets.py             # [MAIN] Reconstruct missing data
├── 🐍 calculate_mad.py                    # [MAIN] Calculate MAD metric
├── 🐍 visualize_mad_comparison.py         # [MAIN] Streamlit dashboard
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
python clean_datasets.py        # Clean and validate raw data
python degrade_datasets.py      # Create degraded datasets
python reconstruct_datasets.py   # Reconstruct missing values
python calculate_mad.py          # Calculate MAD metric
streamlit run visualize_mad_comparison.py  # Visualize results
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
```

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECONSTRUCTION WORKFLOW                       │
└─────────────────────────────────────────────────────────────────┘

0️⃣ DATA CLEANING (clean_datasets.py)
   ├─ Load: data/0_source_data/*.csv (raw data)
   ├─ Auto-detect CSV format (separator, decimal point)
   ├─ Validate & convert: index (datetime/numeric), values (float)
   ├─ Remove: duplicates, invalid rows
   └─ Save: data/1_cleaned_data/*.csv (standardized format)
   
   📊 Example: 6 raw datasets → 6 cleaned datasets

                              ↓

1️⃣ DEGRADATION (degrade_datasets.py)
   ├─ Auto-discover: data/1_cleaned_data/*.csv
   ├─ Apply: MCAR / MAR / MNAR
   ├─ Rates: 10%, 20% (from config.yaml)
   ├─ Iterations: 2x (from config.yaml)
   └─ Save: data/2_missing_data/{dataset}_{technique}_{rate}p_{iter}.csv
   
   📊 Example: 6 datasets × 3 techniques × 2 rates × 2 iterations = 72 files

                              ↓

2️⃣ RECONSTRUCTION (reconstruct_datasets.py)
   ├─ Load: data/2_missing_data/*.csv
   ├─ Apply: 20 reconstruction models (from config.yaml)
   └─ Save: data/3_fixed_data/{dataset}_{technique}_{rate}p_{iter}_{model}.csv
   
   📊 Example: 72 × 20 = 1,440 files

                              ↓

3️⃣ EVALUATION (calculate_mad.py)
   ├─ Load: data/3_fixed_data/*.csv
   ├─ Compare ONLY missing values with: data/1_cleaned_data/*.csv
   ├─ Calculate: MAD (Mean Absolute Difference)
   └─ Save: experiments_results/reconstruction_results_YYYYMMDD_HHMMSS.csv
   
   📊 Output: Single CSV with 1,440 rows
   
   ⚠️  IMPORTANT: MAD is calculated ONLY for values that were missing!

                              ↓

4️⃣ VISUALIZATION (visualize_mad_comparison.py)
   ├─ Load: experiments_results/*.csv
   ├─ Interactive Streamlit dashboard
   ├─ Compare: Models, Techniques, Rates, Datasets
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

## 🎯 Advanced Usage

### Custom Parameters via CLI

```bash
# Degrade specific datasets
python degrade_datasets.py --dataset-files data/0_source_data/boiler.csv

# Use custom config
python degrade_datasets.py --config my_config.yaml

# Override config parameters
python degrade_datasets.py --techniques MCAR --rates 0.05 0.10 --iterations 3

# Reconstruct with specific models
python reconstruct_datasets.py --models interpolate_linear knn

# Calculate with custom config
python calculate_mad.py --config my_config.yaml
```

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
python reconstruct_datasets.py --models my_model
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
python degrade_datasets.py --techniques MY_TECHNIQUE
```

## 🎨 Visualization

The Streamlit dashboard (`visualize_mad_comparison.py`) provides:

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
streamlit run visualize_mad_comparison.py
# Open browser at http://localhost:8501
```

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

### Timing (for 6 datasets)
- **Degradation**: ~5-10 minutes
- **Reconstruction (20 models)**: ~2-4 hours
- **Calculation**: ~10-20 minutes
- **Total Pipeline**: ~3-5 hours

### Disk Space
- **Degraded data**: ~100-500 MB
- **Reconstructed data**: ~2-8 GB
- **Results**: ~1-5 MB per experiment

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
**Last Updated**: December 2024  
**Status**: Production Ready ✅

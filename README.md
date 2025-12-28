# Time Series Reconstruction Framework

A modular framework for evaluating time series reconstruction methods on univariate datasets with various types of missing data.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 📋 Table of Contents

- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
  - [Setup Virtual Environment](#1-setup-virtual-environment-recommended)
  - [Installation](#2-installation)
  - [Configuration](#3-configure-experiment)
  - [Run Pipeline](#4-run-pipeline)
  - [Common Commands](#5-common-commands)
- [⚙️ Configuration](#-configuration)
- [🔄 Workflow](#-workflow)
- [📊 Output Files](#-output-files)
- [📈 Visualization](#-visualization)
- [🎯 Advanced Usage](#-advanced-usage)
  - [Running with tmux](#running-long-experiments-with-tmux)
  - [Custom Parameters](#custom-parameters-via-cli)
  - [Hyperparameter Optimization](#optimizing-stable-diffusion-hyperparameters)
  - [Available Models](#available-models)
- [📝 Adding New Models](#-adding-new-models)
- [🧹 Cleanup](#-cleanup)
- [📈 Performance Estimates](#-performance-estimates)
- [⚠️ Important Notes](#-important-notes)
- [🤝 Citation](#-citation)

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
├── 📁 src/                                 # Source code directory
│   ├── 🐍 1_clean_datasets.py              # [MAIN] Clean and validate raw data
│   ├── 🐍 2_degrade_datasets.py            # [MAIN] Introduce missing data
│   ├── 🐍 3_reconstruct_datasets.py        # [MAIN] Reconstruct missing data
│   ├── 🐍 4_calculate_mad.py               # [MAIN] Calculate MAD metric
│   ├── 🐍 5_visualize_mad_comparison.py    # [MAIN] Streamlit dashboard
│   │
│   ├── 📁 utils/                           # Utility modules
│   │   ├── 🐍 config_loader.py                # Configuration manager
│   │   ├── 🐍 performance_metrics.py          # Performance monitoring
│   │   └── 🐍 statistical_tests.py            # Statistical significance tests
│   │
│   ├── 📁 optimization/                    # Hyperparameter optimization
│   │   └── 🐍 optimize_sd_hyperparams.py      # SD hyperparameter tuning
│   │
│   ├── 📁 reconstruction_models/           # 20 reconstruction models
│   │   ├── 🐍 impute_*.py                     # Simple imputation (mean, median, mode, ffill, bfill)
│   │   ├── 🐍 interpolate_*.py                # Interpolation (linear, cubic, spline, etc.)
│   │   ├── 🐍 knn.py                          # K-Nearest Neighbors
│   │   ├── 🐍 sarimax.py                      # SARIMA with Kalman smoothing
│   │   └── 🐍 stable_diffusion_2_*.py         # Stable Diffusion 2 (GAF, MTF, RP, Spectrogram)
│   │
│   └── 📁 missingness_techniques/          # Missingness patterns
│       ├── 🐍 mcar.py                         # Missing Completely At Random
│       ├── 🐍 mar.py                          # Missing At Random
│       └── 🐍 mnar.py                         # Missing Not At Random
│
├── 📁 data/
│   ├── 📁 0_source_data/                   # Original datasets (auto-discovered)
│   ├── 📁 1_cleaned_data/                  # Cleaned datasets (generated)
│   ├── 📁 2_missing_data/                  # Degraded datasets (generated)
│   └── 📁 3_fixed_data/                    # Reconstructed datasets (generated)
│
├── 📁 experiments_results/                 # Experiment results
│   ├── reconstruction_results_*.csv        # MAD + performance metrics (merged)
│   └── 📁 performance_metrics/             # Performance metrics archive
│       └── performance_metrics_*.csv       # Individual performance logs
│
├── 📁 .streamlit/                          # Streamlit configuration
│   └── config.toml                         # Streamlit settings (file watching disabled)
│
├── ⚙️ config.yaml                          # Main configuration file
├── 📝 Makefile                             # Make commands for easy pipeline execution
├── 📝 requirements.txt                     # Python dependencies
├── 📝 setup_venv.sh                        # Setup script (Linux/Mac)
├── 📝 setup_venv.ps1                       # Setup script (Windows)
├── 📝 LICENSE                              # Apache 2.0 License
└── 📝 README.md                            # This file
```

## 🚀 Quick Start

### 1. Setup Virtual Environment (Recommended)

Using a virtual environment ensures isolated dependencies and prevents conflicts.

#### Quick Setup (Recommended):

**Linux/Mac:**
```bash
# Run setup script (creates experiment env and installs dependencies)
chmod +x setup_venv.sh
./setup_venv.sh
```

**Windows (PowerShell):**
```powershell
# Run setup script (creates experiment env and installs dependencies)
.\setup_venv.ps1

# If you get execution policy error, run first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Manual Setup:

**Linux/Mac:**

```bash
# Create virtual environment
python3 -m venv experiment

# Activate virtual environment
source experiment/bin/activate

# You should see (experiment) in your terminal prompt
```

**Windows (PowerShell):**

```powershell
# Create virtual environment
python -m venv experiment

# Activate virtual environment
.\experiment\Scripts\Activate.ps1

# You should see (experiment) in your terminal prompt
```

**Windows (CMD):**

```cmd
# Create virtual environment
python -m venv experiment

# Activate virtual environment
experiment\Scripts\activate.bat
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

**Key dependencies**:
- `psutil` - CPU and RAM monitoring
- `GPUtil` - GPU monitoring (optional, only if CUDA available)
- `scipy` - Statistical tests (t-tests for model comparison)

If GPU monitoring is not needed, GPUtil can be skipped (performance metrics will still work without it).

### 3. Configure Experiment

Edit `config.yaml` to set:
- Datasets to use (auto-discovers all CSVs in `data/0_source_data/`)
- Reconstruction models to test
- Missingness techniques (MCAR, MAR, MNAR)
- Missing rates (e.g., 10%, 20%)
- Number of iterations

### 4. Run Pipeline

**Using Makefile (Recommended):**

```bash
# See all available commands
make help

# Option 1: Run complete pipeline
make pipeline

# Option 2: Run step by step
make clean-datasets    # Step 1: Clean and validate raw data
make degrade           # Step 2: Create degraded datasets
# OPTIONAL: Optimize Stable Diffusion hyperparameters (run once)
make optimize          # Find optimal num_inference_steps and guidance_scale
# Then update config.yaml with recommended values
make reconstruct       # Step 3: Reconstruct missing values
make calculate         # Step 4: Calculate MAD metric
make visualize         # Step 5: Launch Streamlit dashboard


```

**Manual execution (alternative):**

```bash
python src/1_clean_datasets.py        # Clean and validate raw data
python src/2_degrade_datasets.py      # Create degraded datasets
python src/optimization/optimize_sd_hyperparams.py # OPTIONAL: Optimize Stable Diffusion hyperparameters
python src/3_reconstruct_datasets.py   # Reconstruct missing values
python src/4_calculate_mad.py          # Calculate MAD metric
streamlit run src/5_visualize_mad_comparison.py  # Visualize results
```

**💡 Tip**: For long-running experiments, use `tmux` to keep processes running in background:
```bash
# Start tmux session
tmux new -s experiments

# Run reconstruction (may take hours with Stable Diffusion)
python src/3_reconstruct_datasets.py

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t experiments
```
See [Running Long Experiments with tmux](#running-long-experiments-with-tmux) for details.

### 5. Common Commands

**Using Makefile (Linux/Mac):**

```bash
# View all available commands
make help

# Quick test (limited data, fast)
make test

# Setup from scratch
make setup          # Create virtual environment
make install        # Install dependencies

# Full workflow
make pipeline       # Run steps 1-4 automatically
make visualize      # View results

# Cleanup
make clean          # Remove generated data
make clean-all      # Remove everything including results
```

**Note**: Windows users can use the manual commands or setup WSL (Windows Subsystem for Linux) to use Makefile.

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
   └─ Save metrics: experiments_results/performance_metrics/performance_metrics_YYYYMMDD_HHMMSS.csv
   
   📊 Example: 72 × 20 = 1,440 files + performance_metrics CSV

                              ↓

3️⃣ EVALUATION (4_calculate_mad.py)
   ├─ Load: data/3_fixed_data/*.csv
   ├─ Compare ONLY missing values with: data/1_cleaned_data/*.csv
   ├─ Calculate: MAD (Mean Absolute Difference)
   ├─ Merge: Performance metrics from step 2
   └─ Save: experiments_results/reconstruction_results_YYYYMMDD_HHMMSS.csv
   
   📊 Output: Single CSV with 1,440 rows (MAD + performance metrics)
   
   ⚠️  IMPORTANT: MAD is calculated ONLY for values that were missing!

                              ↓

4️⃣ VISUALIZATION (5_visualize_mad_comparison.py)
   ├─ Load: experiments_results/*.csv (MAD results + performance metrics)
   ├─ Interactive Streamlit dashboard with 11 tabs:
   │  ├─ 📊 By Model: Compare reconstruction quality across models
   │  ├─ 🎯 By Technique: Compare MCAR vs MAR vs MNAR
   │  ├─ 📉 By Missing Rate: Compare 10% vs 20% etc.
   │  ├─ 📁 By Dataset: Compare performance per dataset
   │  ├─ ⚡ By Efficiency: Overall efficiency score (time + CPU + memory)
   │  ├─ 🔥 Heatmap: Overall performance matrix
   │  ├─ 📊 Statistical Tests: Pairwise t-tests (p<0.01, p<0.05) between models
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

### Performance Metrics

**Two locations for performance data:**

1. **`experiments_results/performance_metrics/performance_metrics_YYYYMMDD_HHMMSS.csv`** - Archive of performance metrics
   - Created during reconstruction (step 2)
   - Permanent record of computational performance
   - Includes timestamp for each reconstruction session

2. **`experiments_results/reconstruction_results_YYYYMMDD_HHMMSS.csv`** - Merged results
   - Combines MAD metrics + performance metrics
   - Created during evaluation (step 3)
   - Single file for easy analysis

**Performance Metrics Columns**:
- `dataset_name` - Dataset name
- `technique` - Missingness technique
- `rate_percent` - Missing rate (%)
- `iteration` - Iteration number
- `model` - Reconstruction model name
- `time_seconds` - **Execution time in seconds**
- `cpu_cores_used` - **CPU cores utilized** (e.g., 1.18 = using 1.18 cores)
- `cpu_cores_total` - **Total CPU cores available** (e.g., 4, 8, 16)
- `memory_mb` - **Peak RAM usage (MB)**
- `memory_total_mb` - **Total system RAM available (MB)**
- `gpu_percent` - GPU utilization (%) - *null if GPU not available*
- `gpu_memory_mb` - GPU memory usage (MB) - *null if GPU not available*
- `gpu_memory_total_mb` - **Total GPU memory available (MB)** - *null if GPU not available*
- `timestamp` - When reconstruction was performed

**Complete Merged File Structure** (`reconstruction_results_*.csv`):
```csv
dataset_name,technique,rate_percent,iteration,model,mad,max_diff,min_diff,std_diff,n_missing,n_total,time_seconds,cpu_percent,memory_mb,gpu_percent,gpu_memory_mb
vibration_sensor_S1,MCAR,10,1,interpolate_linear,5.23,12.45,0.12,3.21,21,210,0.15,12.5,45.2,,
vibration_sensor_S1,MCAR,10,1,stable_diffusion_2_gaf,4.87,11.23,0.09,2.98,21,210,45.8,85.3,1024.5,75.2,2048.0
```

**Benefits**:
- **Dual storage** - Archive in `performance_metrics/` + merged in `reconstruction_results`
- **Easy analysis** - Compare quality (MAD) vs. efficiency (time/resources) directly
- **Streamlit dashboard** - Automatically shows performance tabs when data is available
- **Historical tracking** - Keep separate performance logs for each reconstruction session

**How It Works**:

The framework automatically collects performance metrics during reconstruction using the `PerformanceMonitor` class:

```python
from utils.performance_metrics import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Reconstruction happens here
reconstructed = model_func(series)

# Metrics collected automatically
metrics = monitor.stop()
# Returns: {'time_seconds': 1.23, 'cpu_cores_used': 1.18, 'cpu_cores_total': 4, 
#           'memory_mb': 128.5, 'memory_total_mb': 16384, 
#           'gpu_percent': 75.2, 'gpu_memory_mb': 2048, 'gpu_memory_total_mb': 8192}
```

**Interpreting Results**:

*Execution Time:*
- **< 1s** - Fast, suitable for real-time applications
- **1-10s** - Moderate, suitable for batch processing  
- **10-60s** - Slow, suitable for offline analysis
- **> 60s** - Very slow, deep learning models (GPU recommended)

*CPU Usage:*
- **< 1 core** - Light computation, can run multiple instances
- **1-4 cores** - Moderate computation
- **> 4 cores** - Heavy parallel processing

*Memory Usage:*
- **< 100 MB** - Low memory footprint
- **100-500 MB** - Moderate memory usage
- **500-2000 MB** - High memory usage
- **> 2000 MB** - Very high, requires 8GB+ RAM

*GPU Usage:*
- **None/0%** - CPU-only model (interpolation, imputation)
- **> 0%** - GPU-accelerated model (Stable Diffusion)
- **High GPU memory** - Requires powerful GPU (4GB+ VRAM)

**Use Cases**:

*1. Model Selection for Edge Devices:*
```python
# Find fast models with good quality
df = pd.read_csv('experiments_results/reconstruction_results_*.csv')
fast_accurate = df[(df['time_seconds'] < 1.0) & (df['mad'] < 5.0)]
```

*2. Time-Quality Trade-off Analysis:*
```python
# Calculate efficiency: lower MAD per second = better
df['efficiency'] = df['mad'] / df['time_seconds']
best_efficiency = df.groupby('model')['efficiency'].mean().sort_values()
```

*3. Hardware Requirements Planning:*
```python
# Find peak resource usage per model
peak_resources = df.groupby('model').agg({
    'memory_mb': 'max',
    'gpu_memory_mb': 'max',
    'time_seconds': 'mean'
})
```

*4. Cost Optimization (Cloud Computing):*
```python
# Estimate cloud costs (GPU ~$0.90/hour, CPU ~$0.10/hour)
df['cost_per_run'] = df.apply(
    lambda row: (row['time_seconds'] / 3600) * (0.90 if row['gpu_percent'] > 0 else 0.10),
    axis=1
)
```

**Best Practices**:
- Run on representative data (performance varies with dataset size)
- Compare on same hardware (results are hardware-dependent)
- Average multiple runs for reliable timings
- Balance accuracy (MAD) vs. efficiency (time/resources)
- Monitor GPU memory for deep learning models

**Troubleshooting**:
- *GPU metrics show None*: Install `GPUtil` (`pip install GPUtil==1.4.0`) or no GPU available
- *High memory usage*: Large datasets, close other applications
- *Inconsistent timings*: System load, background processes interfering
- *Streamlit "inotify watch limit reached"*: Create `.streamlit/config.toml` with:
  ```toml
  [server]
  fileWatcherType = "none"
  ```
  Or increase system limit: `sudo sysctl fs.inotify.max_user_watches=524288`

**Advanced: Custom Performance Monitoring**

You can use the `PerformanceMonitor` class in your own code:

```python
from utils.performance_metrics import monitor_performance, is_gpu_model

# Monitor your own code
with monitor_performance() as monitor:
    # Your heavy computation here
    result = heavy_computation()

metrics = monitor.stop()
print(f"Time: {metrics['time_seconds']:.2f}s")
print(f"CPU: {metrics['cpu_cores_used']:.2f}/{metrics['cpu_cores_total']} cores")
print(f"RAM: {metrics['memory_mb']:.1f}/{metrics['memory_total_mb']:.0f} MB")

# Check if model uses GPU
if is_gpu_model('stable_diffusion_2_gaf'):
    print("This model will use GPU if available")
```

**Dependencies**:
- `psutil` - Required for CPU and RAM monitoring (included in requirements.txt)
- `GPUtil==1.4.0` - Optional for GPU monitoring (included in requirements.txt)

## 🎯 Advanced Usage

### Running Long Experiments with tmux

For long-running experiments (especially with Stable Diffusion models), use `tmux` to keep the process running even after disconnecting:

```bash
# Start a new tmux session
tmux new -s experiments

# Inside tmux, activate virtual environment and run experiments
source experiment/bin/activate
python src/3_reconstruct_datasets.py

# Detach from tmux session (keeps it running in background)
# Press: Ctrl+B, then D

# Reattach to the session later
tmux attach -t experiments

# List all tmux sessions
tmux ls

# Kill a session when done
tmux kill-session -t experiments
```

**Useful tmux commands**:
- `Ctrl+B` then `D` - Detach from session (process keeps running)
- `Ctrl+B` then `C` - Create new window
- `Ctrl+B` then `N` - Next window
- `Ctrl+B` then `P` - Previous window
- `Ctrl+B` then `[` - Scroll mode (use arrows, `Q` to exit)

**Why use tmux?**:
- Experiments continue running even if SSH disconnects
- Run multiple experiments in parallel windows
- Monitor progress in one window, logs in another
- Return to check progress anytime without interrupting

### Custom Parameters via CLI

For custom parameters, use the scripts directly instead of Makefile:

```bash
# Degrade specific datasets
python src/2_degrade_datasets.py --dataset-files data/0_source_data/boiler.csv

# Use custom config
python src/2_degrade_datasets.py --config my_config.yaml

# Override config parameters
python src/2_degrade_datasets.py --techniques MCAR --rates 0.05 0.10 --iterations 3

# Reconstruct with specific models
python src/3_reconstruct_datasets.py --models interpolate_linear knn

# Calculate with custom config
python src/4_calculate_mad.py --config my_config.yaml
```

**Quick test with Makefile** (limited data for fast testing):

```bash
# Runs: MCAR 10% 1 iteration with only 2 models
make test
```

### Optimizing Stable Diffusion Hyperparameters

The framework includes a dedicated script to find **globally optimal** `num_inference_steps` and `guidance_scale` by testing on **ALL available degraded datasets**:

```bash
# Full optimization (tests on ALL degraded datasets, all 4 SD models)
python src/optimization/optimize_sd_hyperparams.py

# Custom parameter ranges
python src/optimization/optimize_sd_hyperparams.py \
  --steps 5 10 20 30 50 \
  --guidance 3.5 5.0 7.5 10.0

# Quick test (limit to first 5 files)
python src/optimization/optimize_sd_hyperparams.py \
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

## 📈 Visualization

The Streamlit dashboard (`5_visualize_mad_comparison.py`) provides comprehensive analysis across **11 interactive tabs**:

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

### 5. ⚡ By Efficiency
Overall computational efficiency ranking:
- **Efficiency Score**: Combined normalized metric (time + CPU + RAM + GPU)
- **Ascending sort**: Lower score = more efficient (best models at top)
- **Time vs Memory scatter**: Visual efficiency comparison with CPU usage as bubble size

**How Efficiency Score is calculated**:
```
Efficiency Score = Time_norm + CPU_norm + Memory_norm + GPU_norm
```
- **Time_norm**: Normalized execution time (0 to 1)
- **CPU_norm**: CPU cores used / total cores available
- **Memory_norm**: Normalized RAM usage (0 to 1)
- **GPU_norm**: GPU memory used / total GPU memory (0 for CPU-only models)

**Score interpretation**:
- **0-1**: Highly efficient (minimal resources)
- **1-2**: Moderately efficient
- **2-4**: Less efficient (resource-intensive, typically GPU-based models)

**Use cases**:
- Select models for edge devices and embedded systems
- Balance reconstruction quality (MAD) vs. computational cost
- Optimize for deployment scenarios (cloud costs, energy efficiency)
- Quick identification of most efficient models

### 6. 🔥 Heatmap
Matrix visualization showing:
- Model × Technique performance
- Model × Dataset performance
- Sortable by technique or dataset

### 7. 📊 Statistical Tests
Pairwise statistical significance testing:
- **Significance matrix**: n×n matrix showing which model differences are statistically significant
- **Color-coded results**:
  - 🟩 Dark Green `+2`: Row model significantly better (p < 0.01)
  - 🟢 Light Green `+1`: Row model significantly better (p < 0.05)
  - ⬜ White `0`: No significant difference
  - 🔴 Red `-1`: Row model significantly worse (p < 0.05)
  - 🟥 Dark Red `-2`: Row model significantly worse (p < 0.01)
- **Model statistics**: Mean, std, median, min, max for each model
- **Significance summary**: How many models each model is significantly better/worse than
- **P-values matrix**: Detailed p-values for all pairwise comparisons

**Use cases**:
- Determine if performance differences are statistically meaningful
- Identify models that are consistently better/worse across experiments
- Validate that "best" model is significantly better than alternatives
- Scientific reporting: support conclusions with statistical evidence

**Method**: Independent samples t-test on multiple iterations (each iteration = one sample)

### 8. 🏆 Best/Worst
Quick overview of top and bottom performers:
- Best 5 models (lowest MAD)
- Worst 5 models (highest MAD)
- Model comparison bar charts

### 9. ⏱️ Computation Time
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

### 10. 💻 Resource Usage
Monitor hardware resource consumption:
- **CPU usage**: Average and peak CPU utilization per model
- **RAM usage**: Memory consumption per model
- **GPU usage**: GPU utilization for deep learning models (if available)
- **Combined CPU + GPU**: Side-by-side comparison for GPU-based models

**Metrics tracked**:
- CPU cores used / total cores available
- RAM usage in MB / total RAM available
- GPU utilization percentage (optional)
- GPU memory usage in MB / total RAM available (optional)
- Combined view showing both CPU and GPU usage for fair comparison

**Use cases**:
- Identify resource-intensive models
- Compare CPU-only vs GPU-accelerated models
- Optimize for limited hardware
- Plan cloud computing costs

### 11. 📋 Raw Data
Direct access to results:
- Searchable table with all metrics
- Sort by any column
- Download filtered results as CSV
- Full data transparency

### Global vs Local Filters
- **Global filters** (sidebar): Apply to ALL tabs
- **Local filters** (within tabs): Tab-specific refinement

### Launch Dashboard

```bash
streamlit run src/5_visualize_mad_comparison.py
# Open browser at http://localhost:8501
```

**Note**: If you encounter "inotify watch limit reached" error on Linux, the project includes `.streamlit/config.toml` with `fileWatcherType = "none"` to disable file watching. This prevents the error while keeping all functionality (you'll need to manually refresh on code changes).

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
python src/3_reconstruct_datasets.py --models my_model
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
python src/2_degrade_datasets.py --techniques MY_TECHNIQUE
```

## 🧹 Cleanup

### Clean Generated Files

```bash
# Remove generated datasets (keep results and source data)
make clean

# Remove everything including results
make clean-all
```

**What gets cleaned:**
- `make clean`: Removes `data/1_cleaned_data/`, `data/2_missing_data/`, `data/3_fixed_data/`
- `make clean-all`: Removes all of the above + `experiments_results/*.csv`

### Remove Virtual Environment

If you want to completely remove the virtual environment:

```bash
# First, deactivate if active
deactivate

# Remove experiment directory
# Linux/Mac:
rm -rf experiment

# Windows (PowerShell):
Remove-Item -Recurse -Force experiment

# Windows (CMD):
rmdir /s /q experiment
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
source experiment/bin/activate

# Windows (PowerShell)
.\experiment\Scripts\Activate.ps1

# Windows (CMD)
experiment\Scripts\activate.bat
```

You should see `(experiment)` in your terminal prompt when active.

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
  title={Univariate Time Series Reconstruction Framework},
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

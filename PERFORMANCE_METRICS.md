# Performance Metrics Collection - Technical Details

> **Note**: For quick overview and common use cases, see the [Performance Metrics section in README.md](README.md#performance-metrics).
> This document provides technical details for advanced users.

## Overview

The framework automatically collects computational performance metrics during dataset reconstruction to help evaluate the **computational complexity** of different reconstruction algorithms.

## Metrics Collected

For each reconstruction, the following metrics are collected:

| Metric | Description | Unit |
|--------|-------------|------|
| `time_seconds` | Total execution time | seconds |
| `cpu_cores_used` | CPU cores utilized (e.g., 1.18 = using 1.18 cores) | cores |
| `cpu_cores_total` | Total CPU cores available | cores |
| `memory_mb` | Peak RAM usage | MB |
| `memory_total_mb` | Total system RAM available | MB |
| `gpu_percent` | GPU utilization (if available) | % |
| `gpu_memory_mb` | GPU memory usage (if available) | MB |
| `gpu_memory_total_mb` | Total GPU memory available (if available) | MB |

## How It Works

### 1. During Reconstruction (`3_reconstruct_datasets.py`)

The `PerformanceMonitor` class tracks resource usage:

```python
from performance_metrics import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Perform reconstruction
reconstructed = model_func(series)

# Collect metrics
metrics = monitor.stop()
# Returns: {'time_seconds': 1.23, 'cpu_cores_used': 1.18, 'cpu_cores_total': 4, 
#           'memory_mb': 128.5, 'memory_total_mb': 16384, 
#           'gpu_percent': 75.2, 'gpu_memory_mb': 2048, 'gpu_memory_total_mb': 8192, ...}
```

### 2. Automatic Saving

Metrics are automatically saved to:
```
experiments_results/performance_metrics_YYYYMMDD_HHMMSS.csv
```

Example output:
```csv
dataset_name,technique,rate_percent,iteration,model,time_seconds,cpu_cores_used,cpu_cores_total,memory_mb,memory_total_mb,gpu_percent,gpu_memory_mb,gpu_memory_total_mb,timestamp
boiler,MCAR,10,1,interpolate_linear,0.123,0.45,4,45.2,16384,,,,20241227_120000
boiler,MCAR,10,1,stable_diffusion_2_gaf,45.8,3.52,4,1024.5,16384,75.2,2048.0,8192.0,20241227_120000
```

### 3. Visualization

View metrics in Streamlit dashboard:
```bash
streamlit run 5_visualize_mad_comparison.py
```

Navigate to:
- **⏱️ Computation Time** tab - Execution time analysis
- **💻 Resource Usage** tab - CPU, RAM, GPU usage

## Use Cases

### 1. Model Selection Based on Hardware

**Scenario**: Choose fastest models for deployment on edge devices

```python
# Filter models by execution time < 1 second
fast_models = df_perf[df_perf['time_seconds'] < 1.0]['model'].unique()
```

### 2. Cost Optimization

**Scenario**: Estimate cloud computing costs

```python
# Calculate total compute time per model
total_time = df_perf.groupby('model')['time_seconds'].sum()
# GPU models cost ~$0.90/hour, CPU models ~$0.10/hour
```

### 3. Time-Quality Trade-off

**Scenario**: Balance reconstruction quality vs. speed

```python
# Merge MAD results with performance metrics
df_combined = pd.merge(df_mad, df_perf, on=['dataset', 'technique', 'rate_percent', 'iteration', 'model'])

# Calculate efficiency: lower MAD per second = better
df_combined['efficiency'] = df_combined['mad'] / df_combined['time_seconds']
```

### 4. Hardware Requirements Planning

**Scenario**: Plan hardware for large-scale experiments

```python
# Find peak resource usage
max_ram = df_perf['memory_mb'].max()  # Plan RAM capacity
max_gpu_mem = df_perf['gpu_memory_mb'].max()  # Plan GPU memory
```

## Visualization Examples

### Computation Time Tab

![Computation Time Analysis](docs/images/computation_time.png)

Features:
- Average execution time per model
- Time distribution (box plots)
- Time by technique and missing rate
- Detailed statistics table

### Resource Usage Tab

![Resource Usage Analysis](docs/images/resource_usage.png)

Features:
- CPU usage per model
- RAM usage per model
- GPU usage (if available)
- Efficiency score (combined metric)
- Time vs Memory scatter plot

## Dependencies

```bash
pip install psutil  # Required for CPU and RAM monitoring
pip install GPUtil  # Optional for GPU monitoring
```

**Note**: GPUtil is optional. If not installed or no GPU is available, `gpu_percent` and `gpu_memory_mb` will be `None`.

## Interpreting Results

### Execution Time

- **< 1s**: Fast, suitable for real-time applications
- **1-10s**: Moderate, suitable for batch processing
- **10-60s**: Slow, suitable for offline analysis
- **> 60s**: Very slow, deep learning models (GPU recommended)

### CPU Usage

- **< 50%**: Light computation, can run multiple instances in parallel
- **50-80%**: Moderate computation
- **> 80%**: Heavy computation, may bottleneck on CPU cores

### Memory Usage

- **< 100 MB**: Low memory footprint
- **100-500 MB**: Moderate memory usage
- **500-2000 MB**: High memory usage
- **> 2000 MB**: Very high, may require 8GB+ RAM

### GPU Usage

- **None/0%**: CPU-only model (interpolation, imputation)
- **> 0%**: GPU-accelerated model (Stable Diffusion)
- **High GPU memory**: Requires powerful GPU (4GB+ VRAM)

## Best Practices

1. **Run on representative data**: Performance metrics vary with dataset size
2. **Compare on same hardware**: Results are hardware-dependent
3. **Multiple runs**: Execution time can vary; average multiple runs
4. **Consider trade-offs**: Balance accuracy (MAD) vs. efficiency (time/resources)
5. **Monitor GPU memory**: Deep learning models can exhaust VRAM

## Troubleshooting

### GPU metrics show None

- GPUtil not installed: `pip install GPUtil`
- No GPU available: Metrics will remain None (expected)
- CUDA not configured: Install CUDA toolkit

### High memory usage

- Large datasets consume more RAM
- Some models (SARIMAX) require significant memory
- Close other applications to free RAM

### Inconsistent timings

- System load affects measurements
- Background processes can interfere
- Run on dedicated machine for reliable benchmarks

## Advanced: Custom Performance Monitoring

### Monitor your own code

```python
from performance_metrics import monitor_performance

with monitor_performance() as monitor:
    # Your code here
    result = heavy_computation()

metrics = monitor.stop()
print(f"Time: {metrics['time_seconds']:.2f}s")
print(f"CPU: {metrics['cpu_cores_used']:.2f}/{metrics['cpu_cores_total']} cores")
print(f"RAM: {metrics['memory_mb']:.1f}/{metrics['memory_total_mb']:.0f} MB")
```

### Check if model uses GPU

```python
from performance_metrics import is_gpu_model

if is_gpu_model('stable_diffusion_2_gaf'):
    print("This model will use GPU if available")
```

## References

- **psutil documentation**: https://psutil.readthedocs.io/
- **GPUtil documentation**: https://github.com/anderskm/gputil



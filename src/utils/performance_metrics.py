"""
Performance Metrics Collection Module

Collects computational performance metrics during time series reconstruction:
- Execution time
- CPU usage
- RAM usage
- GPU usage (if available)
"""

import time
import psutil
import os
from typing import Dict, Optional
from contextlib import contextmanager


class PerformanceMonitor:
    """Monitor and record performance metrics during reconstruction"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process(os.getpid())
        self.initial_cpu_percent = None
        self.initial_memory_mb = None
        
        # Try to import GPU monitoring
        self.gpu_available = False
        try:
            import GPUtil
            self.GPUtil = GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_available = True
        except ImportError:
            pass
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        
        # Reset CPU percent (first call returns 0)
        self.process.cpu_percent(interval=None)
        
        # Record initial memory
        mem_info = self.process.memory_info()
        self.initial_memory_mb = mem_info.rss / (1024 * 1024)
    
    def stop(self) -> Dict[str, float]:
        """
        Stop monitoring and return collected metrics
        
        Returns:
            Dictionary with performance metrics:
            - time_seconds: Execution time in seconds
            - cpu_percent: Average CPU usage percentage
            - memory_mb: Peak memory usage in MB
            - gpu_percent: GPU utilization percentage (if available)
            - gpu_memory_mb: GPU memory usage in MB (if available)
        """
        self.end_time = time.time()
        
        # Calculate execution time
        execution_time = self.end_time - self.start_time
        
        # Get CPU usage (average since start)
        cpu_percent = self.process.cpu_percent(interval=None)
        
        # Get peak memory usage
        mem_info = self.process.memory_info()
        memory_mb = mem_info.rss / (1024 * 1024)
        peak_memory_mb = memory_mb - self.initial_memory_mb
        
        metrics = {
            'time_seconds': round(execution_time, 3),
            'cpu_percent': round(cpu_percent, 2),
            'memory_mb': round(peak_memory_mb, 2),
            'gpu_percent': None,
            'gpu_memory_mb': None
        }
        
        # Get GPU metrics if available
        if self.gpu_available:
            try:
                gpus = self.GPUtil.getGPUs()
                if gpus:
                    # Use first GPU
                    gpu = gpus[0]
                    metrics['gpu_percent'] = round(gpu.load * 100, 2)
                    metrics['gpu_memory_mb'] = round(gpu.memoryUsed, 2)
            except Exception:
                pass
        
        return metrics


@contextmanager
def monitor_performance():
    """
    Context manager for monitoring performance
    
    Usage:
        with monitor_performance() as monitor:
            # Your code here
            pass
        metrics = monitor.stop()
    """
    monitor = PerformanceMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        pass


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display"""
    lines = [
        f"Time: {metrics['time_seconds']:.2f}s",
        f"CPU: {metrics['cpu_percent']:.1f}%",
        f"RAM: {metrics['memory_mb']:.1f} MB"
    ]
    
    if metrics.get('gpu_percent') is not None:
        lines.append(f"GPU: {metrics['gpu_percent']:.1f}%")
    if metrics.get('gpu_memory_mb') is not None:
        lines.append(f"GPU RAM: {metrics['gpu_memory_mb']:.1f} MB")
    
    return " | ".join(lines)


def is_gpu_model(model_name: str) -> bool:
    """Check if model uses GPU"""
    gpu_keywords = ['stable_diffusion', 'neural', 'deep', 'transformer', 'bert', 'gpt']
    return any(keyword in model_name.lower() for keyword in gpu_keywords)


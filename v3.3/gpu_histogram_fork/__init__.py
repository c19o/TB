"""
savage22-gpu-histogram: GPU-accelerated histogram building for LightGBM sparse training.

Replaces LightGBM's CPU histogram kernel with cuSPARSE/atomic GPU kernels
to eliminate the sparse single-thread bottleneck on 1M+ feature matrices.
"""

from .src.histogram_cusparse import CuSparseHistogramBuilder
from .src.histogram_atomic import AtomicHistogramBuilder
from .src.memory_manager import GPUMemoryManager
from .src.gpu_histogram_wrapper import GPUHistogramBuilder

__version__ = '0.1.0'
__all__ = [
    'CuSparseHistogramBuilder',
    'AtomicHistogramBuilder',
    'GPUMemoryManager',
    'GPUHistogramBuilder',
]

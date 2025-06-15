"""
Memory Utilities for Hybrid Model Training
==========================================

Utilities for managing GPU memory, detecting memory constraints,
and optimizing memory usage during training.
"""

import torch
import gc
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"total_gb": 0.0, "allocated_gb": 0.0, "free_gb": 0.0, "percent_used": 0.0}
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory
    
    return {
        "total_gb": total_memory / (1024**3),
        "allocated_gb": allocated_memory / (1024**3),
        "free_gb": free_memory / (1024**3),
        "percent_used": (allocated_memory / total_memory) * 100
    }

def clear_gpu_cache():
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def log_memory_usage(prefix: str = ""):
    """Log current memory usage."""
    memory_info = get_gpu_memory_info()
    logger.info(f"{prefix}GPU Memory: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB ({memory_info['percent_used']:.1f}%)")

def check_memory_available(required_gb: float = 4.0) -> bool:
    """Check if enough GPU memory is available."""
    memory_info = get_gpu_memory_info()
    return memory_info['free_gb'] >= required_gb

# Simple memory management functions
def optimize_memory():
    """Simple memory optimization."""
    clear_gpu_cache()

def get_memory_efficient_batch_size() -> int:
    """Get a memory-efficient batch size based on available GPU memory."""
    memory_info = get_gpu_memory_info()
    
    if memory_info['total_gb'] >= 16:
        return 16
    elif memory_info['total_gb'] >= 8:
        return 8
    else:
        return 4 
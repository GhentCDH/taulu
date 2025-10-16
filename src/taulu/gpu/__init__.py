"""GPU-accelerated components for taulu.

This module is only available when taulu is installed with GPU support:
    pip install taulu[gpu]
"""

try:
    import torch
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    from . import model, data, train, run
    __all__ = ['GPU_AVAILABLE', 'model', 'data', 'train', 'run']
else:
    __all__ = ['GPU_AVAILABLE']

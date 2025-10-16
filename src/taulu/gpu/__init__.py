"""GPU-accelerated components for taulu.

This module is only available when taulu is installed with GPU support:
    pip install taulu[gpu]
"""

# from . import model, data, train, run

try:
    import PIL
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
if GPU_AVAILABLE:
    __all__ = ['GPU_AVAILABLE', 'model', 'data', 'train', 'run']
else:
    __all__ = ['GPU_AVAILABLE']

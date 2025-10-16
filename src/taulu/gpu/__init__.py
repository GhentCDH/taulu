"""GPU-accelerated components for taulu.

This module is only available when taulu is installed with GPU support:
    pip install taulu[gpu]
"""

try:
    import pillow
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
if GPU_AVAILABLE:
    __all__ = ['GPU_AVAILABLE']
else:
    __all__ = ['GPU_AVAILABLE']

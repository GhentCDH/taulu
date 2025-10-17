"""
Torch based Corner Detection kernel using CNNs.
"""

GPU_AVAILABLE = False

try:
    import torch
    from . import model, data, train, run

    from .model import DeepConvNet
    from .run import apply_kernel_to_image_tiled
    from .train import train_model

    if torch.cuda.is_available():
        GPU_AVAILABLE = True

    __all__ = [
        "GPU_AVAILABLE",
        "model",
        "DeepConvNet",
        "apply_kernel_to_image_tiled",
        "train_model",
        "data",
        "train",
        "run",
    ]

except ImportError:
    __all__ = ["GPU_AVAILABLE"]

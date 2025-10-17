"""
GPU-accelerated corner detection using deep learning.

Requires: `pip install taulu[torch]`

You should install torch separately according to your system.
See [PyTorch installation instructions](https://pytorch.org/get-started/locally/)

Usage:
    >>> from taulu.gpu import DeepConvNet, apply_kernel_to_image_tiled
    >>>
    >>> # Load trained model
    >>> model = DeepConvNet.load("model.pth")
    >>>
    >>> # Generate heatmap for Taulu
    >>> filtered = apply_kernel_to_image_tiled(model, "table.png")
    >>>
    >>> # Use with Taulu
    >>> taulu = Taulu("header.png")
    >>> grid = taulu.segment_table("table.png", filtered=filtered)
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

import logging

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def apply_kernel_to_image_tiled(
    model: nn.Module,
    image: Path | np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Apply trained model to image, producing corner detection heatmap.

    Processes image in tiles to avoid GPU memory issues.

    Args:
        model: Trained DeepConvNet model (call model.eval() first)
        image: Input image path or grayscale numpy array
        device: 'cuda' or 'cpu'
        tile_size: Tile size in pixels (smaller = less memory)
        overlap: Overlap between tiles (must cover receptive field)

    Returns:
        Grayscale uint8 image (0-255) with high values at corners

    Example:
        >>> model = DeepConvNet.load("model.pth")
        >>> filtered = apply_kernel_to_image_tiled(model, "table.png")
        >>> # Use with Taulu
        >>> taulu.segment_table("table.png", filtered=filtered)
    """

    model.eval()
    model = model.to(device)

    # Load or process image based on input type
    if isinstance(image, (Path, str)):
        # Load from file
        img = Image.open(image).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        # Process numpy array
        img_array = image.copy()

        # Handle different input shapes
        if img_array.ndim == 3:
            # If (H, W, 1), squeeze to (H, W)
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(axis=2)
            else:
                raise ValueError(
                    f"Expected grayscale image, got shape {img_array.shape}"
                )
        elif img_array.ndim != 2:
            raise ValueError(f"Expected 2D or 3D array, got shape {img_array.shape}")

        # Normalize to [0, 1] if needed
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)
            # Ensure values are in [0, 1] range
            if img_array.max() > 1.0 or img_array.min() < 0.0:
                img_array = np.clip(img_array, 0, 1)
    else:
        raise TypeError(f"imagemust be Path, str, or np.ndarray, got {type(image)}")

    h, w = img_array.shape

    # Calculate receptive field for overlap
    rf = 1 + len([m for m in model.convs if isinstance(m, nn.Conv2d)]) * (
        model.kernel_size - 1
    )
    overlap = max(overlap, rf)  # Ensure overlap covers receptive field

    # Initialize output heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)  # For averaging overlapping regions

    with torch.no_grad():
        # Process tiles with overlap
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Extract tile with bounds checking
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)

                tile = img_array[y:y_end, x:x_end]

                # Convert to tensor
                tile_tensor = (
                    torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
                )

                # Apply model
                tile_out = model.convs(tile_tensor)
                tile_out = model.conv_final(tile_out)
                tile_heatmap = torch.sigmoid(tile_out).squeeze().cpu().numpy()

                # Calculate valid output region (accounting for padding=0 shrinkage)
                out_h, out_w = tile_heatmap.shape

                # Calculate where this tile's output goes in the full heatmap
                # The output is centered on the input tile
                pad_y = (tile.shape[0] - out_h) // 2
                pad_x = (tile.shape[1] - out_w) // 2

                out_y_start = y + pad_y
                out_x_start = x + pad_x
                out_y_end = out_y_start + out_h
                out_x_end = out_x_start + out_w

                # Accumulate results (for averaging overlaps)
                heatmap[out_y_start:out_y_end, out_x_start:out_x_end] += tile_heatmap
                count_map[out_y_start:out_y_end, out_x_start:out_x_end] += 1

                logger.debug(f"Processed tile ({x}, {y}) -> ({x_end}, {y_end})")

    # Average overlapping regions
    heatmap = np.divide(heatmap, count_map, where=count_map > 0)

    # Ensure C-contiguous array for PyO3 compatibility
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = np.ascontiguousarray(heatmap)

    return heatmap

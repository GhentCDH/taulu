import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image


def apply_kernel_to_image_tiled(
    model: nn.Module,
    image_path: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tile_size: int = 512,
    overlap: int = 64
) -> np.ndarray:
    """
    Apply model in tiles to avoid GPU memory issues.
    """

    model.eval()
    model = model.to(device)

    # Load image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0
    h, w = img_array.shape

    # Calculate receptive field for overlap
    rf = 1 + len([m for m in model.convs if isinstance(m, nn.Conv2d)]) * (model.kernel_size - 1)
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
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)

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

                print(f"Processed tile ({x}, {y}) -> ({x_end}, {y_end})")

    # Average overlapping regions
    heatmap = np.divide(heatmap, count_map, where=count_map > 0)

    return heatmap

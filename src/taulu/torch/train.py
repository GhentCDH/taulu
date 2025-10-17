import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
from pathlib import Path
from os import PathLike
import logging

from .data import IntersectionDataset
from .model import DeepConvNet

_logger = logging.getLogger(__name__)

KERNEL_SIZE = 9
INITIAL_FILTERS = 8
NUM_LAYERS = 7


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str | PathLike = "best_intersection_model.pth",
):
    """Train the model."""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)

            outputs = outputs.squeeze()
            labels = labels.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for patches, labels in val_loader:
                    patches, labels = patches.to(device), labels.to(device)
                    outputs = model(patches)

                    outputs = outputs.squeeze()
                    labels = labels.squeeze()

                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            scheduler.step(val_loss)

            _logger.info(f"Epoch {epoch + 1}/{epochs}")
            _logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            _logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(save_path)
                _logger.info(f"  → Model saved to {save_path}")
        else:
            _logger.info(f"Epoch {epoch + 1}/{epochs}")
            _logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Save model after each epoch if no validation set
            model.save(save_path)

    # Save final model regardless
    final_path = save_path.replace(".pth", "_final.pth")
    model.save(final_path)
    _logger.info(f"Final model saved to {final_path}")
    _logger.info(f""""Use model like this: 
    model = DeepConvNet.load('{final_path}')
    model.eval()")
    """)

    return model


def train_model(
    image_paths: List[Path],
    intersection_coords: List[List[Tuple[int, int]]],
    save_path: str | PathLike = "intersection_mode.pth",
):
    """
    Train corner detection model on annotated images.

    Args:
        image_paths: Paths to training images
        intersection_coords: List of corner coordinates for each image
            Format: [[(x1,y1), (x2,y2), ...], ...] - one list per image
        save_path: Where to save trained model

    Example:
        >>> from pathlib import Path
        >>> import json
        >>>
        >>> # Load annotations from saved TableGrids
        >>> images = [Path("table_01.png"), Path("table_02.png")]
        >>> coords = []
        >>> for img in images:
        >>>     with open(img.with_suffix(".json")) as f:
        >>>         data = json.load(f)
        >>>         corners = [(x,y) for row in data["points"]
        >>>                   for (x,y) in row if x is not None]
        >>>         coords.append(corners)
        >>>
        >>> train_model(images, coords, "my_model.pth")
    """

    dataset = IntersectionDataset(
        image_paths=image_paths,
        intersection_coords=intersection_coords,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    _logger.info("Training deep convolutional model...")
    model = DeepConvNet(
        kernel_size=KERNEL_SIZE, initial_filters=INITIAL_FILTERS, num_layers=NUM_LAYERS
    )
    _train_model(model, train_loader, val_loader, save_path=save_path)

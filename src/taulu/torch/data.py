import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple


class IntersectionDataset(Dataset):
    """Simplified dataset that returns full patches for kernel-based learning."""

    def __init__(
        self,
        image_paths: List[Path],
        intersection_coords: List[List[Tuple[int, int]]],
        patch_size: int = 128,
        negative_samples_per_positive: int = 5,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.intersection_coords = intersection_coords
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.neg_ratio = negative_samples_per_positive
        self.augment = augment

        # Load images
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert("L")
            img = np.array(img, dtype=np.float32) / 255.0
            self.images.append(img)

        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []

        for img_idx, (img, coords) in enumerate(
            zip(self.images, self.intersection_coords)
        ):
            h, w = img.shape

            # Positive samples
            for x, y in coords:
                if (
                    self.half_patch <= x < w - self.half_patch
                    and self.half_patch <= y < h - self.half_patch
                ):
                    samples.append((img_idx, x, y, 1.0))

            # Negative samples
            n_positive = len(coords)
            n_negative = n_positive * self.neg_ratio
            exclusion_radius = self.patch_size // 5

            for i in range(n_negative):
                attempts = 0
                while attempts < 100:
                    x = np.random.randint(self.half_patch, w - self.half_patch)
                    y = np.random.randint(self.half_patch, h - self.half_patch)

                    far_enough = all(
                        abs(x - ix) > exclusion_radius or abs(y - iy) > exclusion_radius
                        for ix, iy in coords
                    )

                    if far_enough:
                        samples.append((img_idx, x, y, 0.0))
                        break
                    attempts += 1

            # Negative samples on lines between intersections
            line_negatives = self._generate_line_negatives(img_idx, coords, h, w)
            samples.extend(line_negatives)

            # ADD: Near-miss negatives (just offset from intersections)
            l = 0
            for x, y in coords:
                for dx, dy in [
                    (10, 0),
                    (-10, 0),
                    (0, 10),
                    (0, -10),
                    (5, 5),
                    (-5, 5),
                    (5, -5),
                    (-5, -5),
                ]:
                    nx, ny = x + dx, y + dy
                    if (
                        self.half_patch <= nx < w - self.half_patch
                        and self.half_patch <= ny < h - self.half_patch
                    ):
                        samples.append((img_idx, nx, ny, 0.0))
                        l += 1

        return samples

    def _generate_line_negatives(
        self, img_idx: int, coords: List[Tuple[int, int]], h: int, w: int
    ) -> List[Tuple[int, int, int, float]]:
        """Generate negative samples on lines between neighboring intersections in the table grid."""
        samples = []

        # Need to reconstruct the row structure from the flattened coords
        # Load the original JSON structure for this image
        image_path = self.image_paths[img_idx]
        # points_path = Path(f"./points_{img_idx}.json")
        points_path = Path("./primitief2.json")

        if not points_path.exists():
            return samples

        import json

        with open(points_path, "r") as f:
            rows = json.load(f)["points"]  # [[[x, y], ...], ...] - list of rows

        # Horizontal lines: sample between neighbors in the same row
        for row in rows:
            for i in range(len(row) - 1):
                x1, y1 = row[i][0], row[i][1]
                x2, y2 = row[i + 1][0], row[i + 1][1]  # Right neighbor in same row

                # Sample 2-3 points between horizontal neighbors
                amount = 6
                for frac in range(1, amount):
                    frac = frac / amount
                    x = int(x1 + (x2 - x1) * frac)
                    y = int(y1 + (y2 - y1) * frac)

                    if (
                        self.half_patch <= x < w - self.half_patch
                        and self.half_patch <= y < h - self.half_patch
                    ):
                        samples.append((img_idx, x, y, 0.0))

        # Vertical lines: sample between neighbors in the same column
        if rows and len(rows) > 0:
            n_cols = len(rows[0])
            for col_idx in range(n_cols):
                # Extract all points in this column across all rows
                column = []
                for row in rows:
                    if col_idx < len(row):
                        column.append(row[col_idx])

                for i in range(len(column) - 1):
                    x1, y1 = column[i][0], column[i][1]
                    x2, y2 = (
                        column[i + 1][0],
                        column[i + 1][1],
                    )  # Bottom neighbor in same column

                    # Sample 2-3 points between vertical neighbors
                    amount = 6
                    for frac in range(1, amount):
                        frac = frac / amount
                        x = int(x1 + (x2 - x1) * frac)
                        y = int(y1 + (y2 - y1) * frac)

                        if (
                            self.half_patch <= x < w - self.half_patch
                            and self.half_patch <= y < h - self.half_patch
                        ):
                            samples.append((img_idx, x, y, 0.0))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, x, y, label = self.samples[idx]
        img = self.images[img_idx]

        # Extract patch
        patch = img[
            y - self.half_patch : y + self.half_patch,
            x - self.half_patch : x + self.half_patch,
        ]

        # Optional augmentation: 90-degree rotations
        if self.augment and np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k)

        patch = torch.from_numpy(patch.copy()).unsqueeze(0)
        label = torch.tensor([label], dtype=torch.float32)

        return patch, label

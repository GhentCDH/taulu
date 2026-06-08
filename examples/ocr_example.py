"""
OCR example: send the cells of a segmented table to TrOCR.

The workflow is:

1. Segment a table image with Taulu to get a `SegmentedTable`.
2. Crop every cell out of the source image with `grid.crop_cell()`
   (perspective-corrected).
3. Batch the crops through a TrOCR model.
4. Write the transcriptions to a CSV that mirrors the table's grid structure.

For a real pipeline you'd usually run a text-detection step first and only
transcribe the cells that actually contain handwriting -- TrOCR is expensive,
so skipping the empty cells is a big speedup. Taulu ships a rough
`grid.text_regions()` helper, but it relies on a simple heuristic and isn't
robust; plug in a proper text detector here instead. For clarity this example
just transcribes every cell.

This example is self-contained: it loads TrOCR directly from `transformers`
so it can be run from the taulu repo alone. It requires `transformers`,
`torch` and `Pillow` in addition to taulu's normal dependencies:

    uv run --with transformers --with torch --with pillow \
        python examples/ocr_example.py

The first run downloads the model weights (~1.3 GB) from the Hugging Face Hub.
"""

import csv
from pathlib import Path

import cv2
from cv2.typing import MatLike
from PIL import Image

from taulu import Split, Taulu
from taulu.grid import SegmentedTable

# A French handwriting fine-tune of TrOCR. The processor (image preprocessing)
# comes from the base handwritten model; the encoder-decoder weights and
# tokenizer are the French fine-tune.
TROCR_PROCESSOR_MODEL = "microsoft/trocr-large-handwritten"
TROCR_MODEL = "agomberto/trocr-large-handwritten-fr"


class TrOCR:
    """Minimal standalone TrOCR wrapper that transcribes OpenCV (BGR) images."""

    def __init__(self):
        import torch
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            logging,
        )

        logging.set_verbosity_error()

        # pick the best available device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        print(f"Loading TrOCR on {self._device}...")

        self._processor = TrOCRProcessor.from_pretrained(TROCR_PROCESSOR_MODEL)
        self._model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
        self._model = self._model.to(self._device)

    def transcribe(self, imgs: list[MatLike]) -> list[str]:
        """Transcribe a batch of BGR cell images to text."""
        if not imgs:
            return []

        # TrOCR expects RGB PIL images
        pil_imgs = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs]

        pixel_values = self._processor(
            images=pil_imgs, return_tensors="pt"
        ).pixel_values.to(self._device)

        generated = self._model.generate(pixel_values)
        return list(self._processor.batch_decode(generated, skip_special_tokens=True))


def ocr_table(grid: SegmentedTable, image: MatLike, ocr: TrOCR) -> list[list[str]]:
    """
    Transcribe every cell of the table, row by row.

    Returns a 2D list of strings mirroring the table's grid (no header row).
    """
    transcriptions: list[list[str]] = []

    for row in range(grid.rows):
        # crop_cell() handles the perspective correction for each cell
        crops = [grid.crop_cell(image, (row, col)) for col in range(grid.cols)]

        # batch the whole row through TrOCR in a single forward pass
        texts = ocr.transcribe(crops)
        transcriptions.append(texts)

        for col, text in enumerate(texts):
            print(f"row {row}, col {col}: {text!r}")

    return transcriptions


def write_csv(transcriptions: list[list[str]], path: str):
    """Write the transcribed cells to a CSV matching the table structure."""
    with open(path, "w, newline="") as f:
        writer = csv.writer(f)
        writer.writerows(transcriptions)
    print(f"Wrote {path}")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    table_image = str(data_dir / "table_00.png")

    # This is a two-page table, so segment with a left/right header Split.
    # For a single-page table, pass a single header template path instead.
    taulu = Taulu(
        Split(
            str(data_dir / "header_left_00.png"),
            str(data_dir / "header_right_00.png"),
        ),
        binarization_sensitivity=0.25,
        extrapolation_distance=30,
        line_gap_fill=7,
        intersection_kernel_size=35,
        search_radius=30,
        row_height_factor=0.85,
        min_rows=45,
    )

    print(f"Segmenting {table_image}...")
    grid = taulu.segment_table(table_image)
    print(f"Detected {grid.rows} rows x {grid.cols} cols")

    image = cv2.imread(table_image)

    ocr = TrOCR()
    transcriptions = ocr_table(grid, image, ocr)

    write_csv(transcriptions, str(output_dir / "table_00.csv"))


if __name__ == "__main__":
    main()

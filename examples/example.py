from taulu import Taulu
import os


def setup():
    # create an Annotation file of the headers in the image
    # (one for the left header, one for the right)
    # and store them in the examples directory
    print("Annotating the LEFT header...")
    Taulu.annotate("../data/table_00.png", "table_00_header_left.png")

    print("Annotating the RIGHT header...")
    Taulu.annotate("../data/table_00.png", "table_00_header_right.png")


def main():
    taulu = Taulu(("table_00_header_left.png", "table_00_header_right.png"))
    table = taulu.segment_table(
        "../data/table_00.png", cell_height_factor=0.8, debug_view=True
    )

    table.show_cells("../data/table_00.png")


if __name__ == "__main__":
    if os.path.exists("table_00_header_left.png") and os.path.exists(
        "table_00_header_right.png"
    ):
        main()
    else:
        setup()
        main()

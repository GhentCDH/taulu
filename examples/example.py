import os

from cv2 import imread
from taulu.img_util import show
from taulu import HeaderAligner, HeaderTemplate, GridDetector


def main(template):
    aligner = HeaderAligner("header.png")
    filter = GridDetector(
        # these are the most important parameters to tune
        kernel_size=41,
        cross_width=10,
        morph_size=7,
        region=60,
        k=0.45,
    )

    # crop the input image (this step is only necessary if the image contains more than just the table)
    table = imread("table.png")
    h = aligner.align(table)

    # find the intersections of rules in the image
    # the `True` parameter means that intermediate results are shown too, for debugging and parameter tuning
    filtered = filter.apply(table, True)
    show(filtered)

    # define the start point
    # note that this skips the left-most column, but we'll add this back in later
    left_top_template = template.intersection((1, 1))
    left_top_template = (int(left_top_template[0]), int(left_top_template[1]))
    left_top_table = aligner.template_to_img(h, left_top_template)

    table_structure = filter.find_table_points(
        table,
        left_top_table,
        # 1 because we skipped the first column with the starting point definition
        template.cell_widths(1),
        # 0.8 means that the height of the cells is approx. 0.8 times the height of the header
        template.cell_height(0.8),
    )

    # add the left column again
    table_structure.add_left_col(template.cell_width(0))

    # you can click the cells to verify that the algorithm works
    table_structure.show_cells(table)


def setup():
    # annotate your header
    return HeaderTemplate.annotate_image("table.png", crop="header.png")


if __name__ == "__main__":
    if not os.path.exists("table.png"):
        print("You need to supply your own table image (table.png)")
        exit(1)

    template = setup()
    main(template)

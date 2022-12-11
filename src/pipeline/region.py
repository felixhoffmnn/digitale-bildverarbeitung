import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Add docstring
def region_of_interest(img_gray: cv.Mat, kitti: bool) -> tuple[cv.Mat, cv.Mat]:
    width = img_gray.shape[1]
    height = img_gray.shape[0]

    # Find region of interest
    if kitti:
        vertices = np.array(
            [
                (width * (1 / 4), height * (1 / 2)),  # Top-left corner
                (0, height - 1),  # Bottom-left corner
                ((width - 1), height - 1),  # Bottom-right corner
                (width * (3 / 4), height * (1 / 2)),  # Top-right corner
            ],
            np.float32,
        )
    else:
        vertices = np.array(
            [
                (width * (2 / 5), height * (13 / 20)),  # Top-left corner
                (0.0, (height - 1) * (9 / 10)),  # Bottom-left corner
                (width - 1, (height - 1) * (9 / 10)),  # Bottom-right corner
                (width * (3 / 5), height * (13 / 20)),  # Top-right corner
            ],
            np.int32,
        )

    # defining a blank mask to start with
    mask = np.zeros_like(img_gray)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv.fillPoly(mask, np.array([vertices], np.int32), [255])

    # returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img_gray, mask)

    return masked_image, vertices

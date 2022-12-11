import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Add docstring
def region_of_interest(img_gray: cv.Mat, kitti: bool = False) -> cv.Mat:
    width = img_gray.shape[1]
    height = img_gray.shape[0]

    # Find region of interest
    if kitti:
        vertices = np.array(
            [
                (width * (1 / 4), height * (1 / 2)),
                (0, height - 1),
                ((width - 1), height - 1),
                (width * (3 / 4), height * (1 / 2)),
            ],
            np.float32,
        )
    else:
        vertices = np.array(
            [
                [140, height],
                [610, height * 0.60],
                [670, height * 0.60],
                [1200, height],
            ],
            np.int32,
        )

    # defining a blank mask to start with
    mask = np.zeros_like(img_gray)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv.fillPoly(mask, np.array([vertices], np.int32), [255])

    # returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img_gray, mask)

    return masked_image

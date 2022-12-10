import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Add docstring
def region_of_interest(img: cv.Mat) -> cv.Mat:
    # Find region of interest
    vertices = np.array(
        [
            [140, img.shape[0]],
            [400, img.shape[0] * 0.75],
            [610, img.shape[0] * 0.6],
            [670, img.shape[0] * 0.6],
            [920, img.shape[0] * 0.75],
            [1200, img.shape[0]],
        ],
        np.int32,
    )

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv.fillPoly(mask, np.int32([vertices]), [255])

    # returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img, mask)

    return masked_image

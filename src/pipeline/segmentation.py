import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Add docstring
def segment_img(img: cv.Mat) -> cv.Mat:
    # Find region of interest
    vertices = np.array(
        [
            [130, img.shape[0]],
            [625, img.shape[0] * 0.6],
            [700, img.shape[0] * 0.6],
            [1200, img.shape[0]],
        ],
        np.int32,
    )

    # defining a blank mask to start with
    mask = np.zeros_like(img)
    ignore_mask_color = (255,)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv.fillPoly(mask, np.int32([vertices]), ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img, mask)

    return masked_image

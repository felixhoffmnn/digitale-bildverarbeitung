import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Add docstring
def transform_perspective(img: cv.Mat) -> cv.Mat:
    src = np.float32([[560, 460], [180, 690], [1130, 690], [750, 460]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

    M = cv.getPerspectiveTransform(src, dst)
    # Minv = cv.getPerspectiveTransform(dst, src)

    transformed_img = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)

    return transformed_img

import sys

import cv2 as cv
import numpy as np
from loguru import logger

# def get_inner_line(img_rgb: cv.Mat, img_hsv: cv.Mat, img_hls: cv.Mat, img_gray: cv.Mat) -> cv.Mat:
#     # Create multiple binary masks for the yellow and white colors
#     b_rgb_thresh = cv.threshold(img_rgb[:, :, 0], 205, 255, cv.THRESH_BINARY)[1]
#     w_gray_thresh = cv.inRange(img_gray, np.array([210]), np.array([255]))
#     y_hsv_thresh = cv.inRange(img_hsv, np.array([15, 35, 35]), np.array([95, 255, 255]))
#     inner_line = cv.bitwise_or(w_gray_thresh, y_hsv_thresh)
#     inner_line = cv.bitwise_or(w_gray_thresh, b_rgb_thresh)

#     return inner_line


def get_outer_mask(img_rgb: cv.Mat, img_lab: cv.Mat) -> tuple[cv.Mat, cv.Mat]:
    w_mask = cv.inRange(img_rgb[:, :, 0], np.array([185]), np.array([255]))
    y_mask = cv.inRange(img_lab, np.array([125, 105, 85]), np.array([200, 135, 115]))

    return w_mask, y_mask


def get_sobel(img: cv.Mat, orient: str = "x", sobel_kernel: int = 3) -> cv.Mat:
    # Apply sobel filtering
    if orient == "x":
        sobel = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobel = cv.convertScaleAbs(sobel)
    elif orient == "y":
        sobel = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = cv.convertScaleAbs(sobel)
    else:
        logger.error("Invalid orientation for sobel filtering")
        sys.exit(1)

    return abs_sobel


# TODO: Refactor this function
# TODO: Add docstring
def thresh_img(img_rgb: cv.Mat) -> cv.Mat:
    img_hls = cv.cvtColor(img_rgb, cv.COLOR_RGB2HLS)
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_gray = cv.equalizeHist(img_gray)

    # outer_mask = get_outer_mask(img_rgb, img_hls)
    # inner_line = get_inner_line(img_rgb, img_hsv, img_hls, img_gray)

    sobel_x = get_sobel(img_gray, "x", 3)
    sobel_y = get_sobel(img_gray, "y", 3)

    sobel = cv.addWeighted(sobel_x, 1, sobel_y, 0.1, 0)
    sobel = cv.threshold(sobel, 30, 255, cv.THRESH_TOZERO)[1]

    w_mask, y_mask = get_outer_mask(img_rgb, img_lab)

    outer_line = cv.threshold(sobel, 35, 255, cv.THRESH_TOZERO)[1]
    w_line = cv.bitwise_and(sobel, w_mask)
    y_line = cv.bitwise_and(sobel, y_mask)
    outer_line = cv.bitwise_or(w_line, y_line)
    outer_line = cv.morphologyEx(outer_line, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    # Convert to binary
    outer_line = cv.threshold(outer_line, 0, 255, cv.THRESH_BINARY)[1]

    return outer_line

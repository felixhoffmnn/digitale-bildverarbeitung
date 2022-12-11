import sys

import cv2 as cv
import numpy as np
from loguru import logger


def get_inner_line(img_rgb: cv.Mat, img_hls: cv.Mat, img_hsv: cv.Mat) -> cv.Mat:
    """Mask the base image for red, white and yellow lines

    Parameters
    ----------
    img_rgb : cv.Mat
        The base image in RGB color space
    img_hls : cv.Mat
        The base image in HLS color space
    img_hsv : cv.Mat
        The base image in HSV color space

    Returns
    -------
    cv.Mat
        The mask for the base image for inner lines
    """
    w_gray_thresh = cv.inRange(img_rgb[:, :, 0], np.array([200]), np.array([255]))
    y_lab_thresh = cv.inRange(img_hsv, np.array([50, 65, 75]), np.array([110, 165, 255]))
    s_hls_thresh = cv.inRange(img_hls[:, :, 2], np.array([180]), np.array([200]))
    s_hsv_thresh = cv.inRange(img_hsv[:, :, 1], np.array([200]), np.array([215]))

    inner_line = cv.bitwise_or(w_gray_thresh, y_lab_thresh)
    inner_line = cv.bitwise_or(inner_line, s_hls_thresh)
    inner_line = cv.bitwise_or(inner_line, s_hsv_thresh)

    return inner_line


def get_sobel(
    img: cv.Mat, orient: str = "x", sobel_kernel: int = 3, sobel_threshold: tuple[int, int] = (40, 100)
) -> cv.Mat:
    """Apply sobel filtering to the image, scale the result and threshold it

    Parameters
    ----------
    img : cv.Mat
        The base image ()
    orient : str, optional
        _description_, by default "x"
    sobel_kernel : int, optional
        _description_, by default 3
    sobel_threshold : tuple[int, int], optional
        _description_, by default (40, 100)

    Returns
    -------
    cv.Mat
        _description_
    """
    if orient == "x":
        sobel = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == "y":
        sobel = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        logger.error("Invalid orientation for sobel filtering")
        sys.exit(1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel >= sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 255

    r_rgb_mask = cv.inRange(img, np.array([140]), np.array([255]))
    s_binary = cv.bitwise_and(s_binary, r_rgb_mask)

    return s_binary


# TODO: Refactor this function
# TODO: Add docstring
def thresh_img(img_rgb: cv.Mat, kitti: bool) -> cv.Mat:
    """First convert the base image to HLS, HSV and grayscale color space.
    Then apply sobel filtering to the grayscale image and mask the base image for red, white and yellow lines.
    Finally combine the two masks and return the result.

    Parameters
    ----------
    img_rgb : cv.Mat
        The base image in RGB color space
    kitti : bool
        If the base image is from the KITTI dataset

    Returns
    -------
    cv.Mat
        The mask for the base image
    """
    img_hls = cv.cvtColor(img_rgb, cv.COLOR_RGB2HLS)
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    img_gray = cv.equalizeHist(img_gray)

    inner_line = get_inner_line(img_rgb, img_hls, img_hsv)
    outer_line = get_sobel(img_gray, "x", 3)

    lane = cv.bitwise_or(inner_line, outer_line)

    if kitti:
        g_rgb_mask = cv.inRange(img_rgb[:, :, 0], np.array([195]), np.array([255]))
        lane = cv.bitwise_and(lane, g_rgb_mask)

    return lane


# calculate light channel mean 65 < 140
# for
# Seperate image into 4 parts (np.concatenate)
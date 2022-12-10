import math

import cv2 as cv
import numpy as np
from cv2 import threshold


# TODO: Refactor this function
# TODO: Fix operations (typing)
# TODO: Add docstring
def thresh_img(img_rgb: cv.Mat) -> cv.Mat:
    # TODO: Enlarge the color ranges
    img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    img_hls = cv.cvtColor(img_rgb, cv.COLOR_RGB2HLS)
    # Isolate the blue channel for sobel filtering, so the yellow line is easily detected
    blue_channel = img_rgb[:, :, 2]

    # Create multiple binary masks for the yellow and white colors
    yellow_hsv_mask = cv.inRange(img_hsv, np.array([0, 75, 95]), np.array([179, 255, 255]))
    white_hls_mask = cv.inRange(img_hls, np.array([0, 205, 0]), np.array([130, 255, 255]))
    lane_mask = cv.inRange(blue_channel, np.array([165]), np.array([225]))
    color_mask = cv.bitwise_or(yellow_hsv_mask, white_hls_mask)
    color_mask = cv.bitwise_or(color_mask, lane_mask)
    # color_mask = cv.bitwise_or(yellow_hsv_mask, white_hls_mask)
    # color_mask = cv.bitwise_or(yellow_rgb_mask, white_rgb_mask)
    # color_mask[color_mask > 0] = 255

    # Apply sobel filteringq
    sobel_x = cv.Sobel(blue_channel, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(blue_channel, cv.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv.convertScaleAbs(sobel_x)
    abs_sobel_y = cv.convertScaleAbs(sobel_y)

    sobel = cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 10)
    sobel[sobel < 20] = 0

    _, s_binary = cv.threshold(img_hls[:, :, 2], 60, 255, cv.THRESH_BINARY)
    _, r_thresh = cv.threshold(img_rgb[:, :, 2], 100, 255, cv.THRESH_BINARY)
    _, l_thresh = cv.threshold(img_hls[:, :, 1], 200, 255, cv.THRESH_BINARY)
    rs_binary = cv.bitwise_and(s_binary, r_thresh)
    rsl_binary = cv.bitwise_or(rs_binary, l_thresh)

    ### Combine the possible lane lines with the possible lane line edges #####
    # If you show rs_binary visually, you'll see that it is not that different
    # from this return value. The edges of lane lines are thin lines of pixels.
    # # Denoise the image
    # sobel[sobel < 15] = 0

    # Combine the binary masks
    # combined_mask = cv.bitwise_or(color_mask, sobel)

    # return cv.bitwise_and(rsl_binary, sobel_color_mask)
    return cv.bitwise_and(color_mask, sobel)


def get_line_markings(frame):
    """
    Isolates lane lines.

        :param frame: The camera frame that contains the lanes we want to detect
    :return: Binary (i.e. black and white) image containing the lane lines.
    """
    # Convert the video frame from BGR (blue, green, red)
    # color space to HLS (hue, saturation, lightness).
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

    ################### Isolate possible lane line edges ######################

    # Perform Sobel edge detection on the L (lightness) channel of
    # the image to detect sharp discontinuities in the pixel intensities
    # along the x and y axis of the video frame.
    # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
    # Relatively light pixels get made white. Dark pixels get made black.
    _, sxbinary = cv.threshold(hls[:, :, 1], 128, 255, cv.THRESH_BINARY)
    sxbinary = cv.GaussianBlur(sxbinary, (3, 3), 0)  # Reduce noise

    # 1s will be in the cells with the highest Sobel derivative values
    # (i.e. strongest lane line edges)
    sxbinary = np.sqrt(
        np.absolute(cv.Sobel(sxbinary, cv.CV_64F, 1, 0, ksize=3)) ** 2
        + np.absolute(cv.Sobel(sxbinary, cv.CV_64F, 0, 1, ksize=3)) ** 2
    )

    ######################## Isolate possible lane lines ######################

    # Perform binary thresholding on the S (saturation) channel
    # of the video frame. A high saturation value means the hue color is pure.
    # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
    # and have high saturation channel values.
    # s_binary is matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the purest hue colors (e.g. >80...play with
    # this value for best results).
    s_channel = hls[:, :, 2]  # use only the saturation channel data
    _, s_binary = cv.threshold(s_channel, 80, 255, cv.THRESH_BINARY)

    # Perform binary thresholding on the R (red) channel of the
    # original BGR video frame.
    # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the richest red channel values (e.g. >120).
    # Remember, pure white is bgr(255, 255, 255).
    # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
    _, r_thresh = cv.threshold(frame[:, :, 0], 120, 255, cv.THRESH_BINARY)

    # Lane lines should be pure in color and have high red channel values
    # Bitwise AND operation to reduce noise and black-out any pixels that
    # don't appear to be nice, pure, solid colors (like white or yellow lane
    # lines.)
    rs_binary = cv.bitwise_and(s_binary, r_thresh)

    ### Combine the possible lane lines with the possible lane line edges #####
    # If you show rs_binary visually, you'll see that it is not that different
    # from this return value. The edges of lane lines are thin lines of pixels.
    return cv.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

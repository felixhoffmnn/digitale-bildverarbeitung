import sys
from typing import Any

import cv2 as cv
import numpy as np
from loguru import logger
from numpy.linalg import LinAlgError

from src.pipeline.line import Line


def lane_sliding_windows(
    img: cv.Mat,
    line: Line,
    nonzero: tuple[np.ndarray[Any, Any], ...],
    n_windows: int,
    window_height: int,
    margin: int,
    h: int,
    x_current: np.int_,
    minpix: int = 25,
) -> tuple[Any, Any, bool]:
    """Gets the new line on the given birdeye image by new sliding windows

    Parameters
    ----------
    img : cv.Mat
        Input image in birdeye perspective
    line : Line
        Line on which the sliding window is to be applied to
    nonzero : tuple[np.ndarray[Any, Any], ...]
        All the nonzero pixels of the input image
    n_windows : int
        Number of sliding windows used to search for the lines
    window_height : int
        Height of the windows
    margin : int
        Helps Calculate window boundaries
    h : int
        Height of the input image
    x_current : np.int_
        Used to calculate window boundaries
    minpix : int, optional
        Boundary which decides if window is recentered, by default 25

    Returns
    -------
    tuple[Any, Any, bool]
        Indices of the lane, Coefficients of the curve, Boolean if lane was detected
    """

    lane_inds = []

    nonzero_y, nonzero_x = nonzero

    # For each sliding window
    for window in range(n_windows):
        # Get boundaries of the window
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        # Draw the windows on the visualization image
        cv.rectangle(img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_inds = (
            (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
        ).nonzero()[0]

        # Append these indices to the lists
        lane_inds.append(good_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int_(np.mean(nonzero_x[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Reset line squared error count
    line.error_count = 0

    # Extract left and right line pixel positions
    line.all_x, line.all_y = nonzero_x[lane_inds], nonzero_y[lane_inds]  # type: ignore

    detected = True
    # If no pixels were found, use the last fit
    if not list(line.all_x) or not list(line.all_y):  # type: ignore
        fit_pixel = line.last_fit_pixel
        detected = False
    else:
        fit_pixel = np.polyfit(line.all_y, line.all_x, 2)  # type: ignore

    return lane_inds, fit_pixel, detected


def get_fits_by_sliding_windows(
    img_birdeye: cv.Mat, line_lt: Line, line_rt: Line, n_windows=9
) -> tuple[cv.Mat, Line, Line]:
    """Gets the new lines on the given birdeye image by new sliding windows

    Parameters
    ----------
    img_birdeye : cv.Mat
        The birdeye image for which the lines are to be detected
    line_lt : Line
        The current left line
    line_rt : Line
        The current right line
    n_windows : int, optional
        The number of windows, by default 9

    Returns
    -------
    tuple[cv.Mat, Line, Line]
        The output image with the new lines drawn on it, the new left line and the new right line
    """
    h, w = img_birdeye.shape

    off_x = w * 0.04

    # Histogram of the bottom half of the image
    histogram = np.sum(img_birdeye[h // 2 : -15, :], axis=0)

    # Prepare the output image
    out_img = np.dstack((img_birdeye, img_birdeye, img_birdeye))

    midpoint = len(histogram) // 2

    # Peak of the left and right halves of the histogram (but off_x pixels from the midpoint)
    leftx_base = np.argmax(histogram[: int(midpoint - off_x)])
    rightx_base = np.argmax(histogram[int(midpoint + off_x) :]) + midpoint

    window_height = int(h / n_windows)

    # x and y positions of all nonzero pixels in the image
    nonzero: tuple[np.ndarray[Any, Any], ...] = img_birdeye.nonzero()
    nonzero_y, nonzero_x = nonzero

    margin = 50  # width of the windows +/- margin

    # Get new indices and coefficients for the left and right lines by sliding windows
    left_lane_inds, left_fit_pixel, left_detected = lane_sliding_windows(
        img_birdeye, line_lt, nonzero, n_windows, window_height, margin, h, leftx_base
    )
    right_lane_inds, right_fit_pixel, right_detected = lane_sliding_windows(
        img_birdeye, line_rt, nonzero, n_windows, window_height, margin, h, rightx_base
    )

    # Update line objects
    line_lt.update_line(left_fit_pixel, detected=left_detected)
    line_rt.update_line(right_fit_pixel, detected=right_detected)

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    return out_img, line_lt, line_rt


def get_fit_by_previous_line(
    name: str, line: Line, nonzero: tuple[np.ndarray[Any, Any], ...], margin: int, h: int
) -> tuple[Any, Any, bool]:
    """Calculates the fit of the line by the previous line.

    This function represents a abstraction, to reduce code duplication.

    Parameters
    ----------
    name : str
        The name of the line
    line : Line
        The line object for which the fit is to be calculated
    nonzero : tuple[np.ndarray[Any, Any], ...]
        Nonzero pixels in the image
    margin : int
        Margin for the fit
    h : int
        Height of the image

    Returns
    -------
    tuple[Any, Any, bool]
        Indices of the pixels, fit of the line, if the line was detected
    """
    nonzero_y, nonzero_x = nonzero

    fit_pixel = line.last_fit_pixel

    # x and y positions of all nonzero pixels in the image
    lane_inds = (nonzero_x > (fit_pixel[0] * (nonzero_y**2) + fit_pixel[1] * nonzero_y + fit_pixel[2] - margin)) & (  # type: ignore
        nonzero_x < (fit_pixel[0] * (nonzero_y**2) + fit_pixel[1] * nonzero_y + fit_pixel[2] + margin)  # type: ignore
    )

    # Extract left and right line pixel positions
    line.all_x, line.all_y = nonzero_x[lane_inds], nonzero_y[lane_inds]

    # If no pixels were found, use the last fit
    detected = True
    if not list(line.all_x) or not list(line.all_y):  # type: ignore
        fit_pixel = line.last_fit_pixel
        detected = False
    else:
        try:
            fit_pixel = np.polyfit(line.all_y, line.all_x, 2)  # type: ignore
        except LinAlgError:
            fit_pixel = line.last_fit_pixel
            detected = False

    # Calculate MSE to check if the fit is valid or differs too much from the last fit
    mean_squared_error = calculate_mean_squared_error(fit_pixel, line.last_fit_pixel, h)  # type: ignore

    # If the MSE is too high, use the last fit
    if mean_squared_error > 205:
        logger.debug(f"{name}_mean_squared_error: {mean_squared_error}")
        fit_pixel = line.last_fit_pixel
        line.detected = False
        line.error_count += 1

    return lane_inds, fit_pixel, detected


def get_fits_by_previous_fits(img_birdeye: cv.Mat, line_lt: Line, line_rt: Line) -> tuple[cv.Mat, Line, Line]:
    """Searches for the lines in the birdeye image by using the previous fits.

    Parameters
    ----------
    img_birdeye : cv.Mat
        The birdeye image in which the lines are to be searched
    line_lt : Line
        The current left line
    line_rt : Line
        The current right line

    Returns
    -------
    tuple[cv.Mat, Line, Line]
        The output image, the updated left line and the updated right line
    """
    h, w = img_birdeye.shape

    # Gets the last polynomial coefficients
    left_fit_pixel = line_lt.last_fit_pixel
    right_fit_pixel = line_rt.last_fit_pixel

    # Set the width of the windows +/- margin
    nonzero = img_birdeye.nonzero()
    margin = 50
    n_windows = 9
    window_height = int(h / n_windows)

    histogram = np.sum(img_birdeye[h // 2 : -15, :], axis=0)

    midpoint = len(histogram) // 2

    off_x = w * 0.04

    # Peak of the left and right halves of the histogram (but off_x pixels from the midpoint)
    leftx_base = np.argmax(histogram[: int(midpoint - off_x)])
    rightx_base = np.argmax(histogram[int(midpoint + off_x) :]) + midpoint

    # Calculate the sum of the pixels (i.e. how light it is) in the image
    sum_light = np.sum(img_birdeye)

    # If there were too many errors in the last fits and the image is not too dark, refit the lines by sliding windows
    if line_lt.error_count > 30 and sum_light > 200000:  # type : ignore
        logger.info(f"Refitting left lane")
        left_lane_inds, left_fit_pixel, left_detected = lane_sliding_windows(
            img_birdeye, line_lt, nonzero, 9, 80, margin, h, leftx_base
        )
    else:
        left_lane_inds, left_fit_pixel, left_detected = get_fit_by_previous_line("left", line_lt, nonzero, margin, h)

    if line_rt.error_count > 30 and sum_light > 200000:  # type : ignore
        logger.info(f"Refitting right lane")
        right_lane_inds, right_fit_pixel, right_detected = lane_sliding_windows(
            img_birdeye, line_rt, nonzero, n_windows, window_height, margin, h, rightx_base
        )
    else:
        right_lane_inds, right_fit_pixel, right_detected = get_fit_by_previous_line(
            "right", line_rt, nonzero, margin, h
        )

    # Update the line objects
    line_lt.update_line(left_fit_pixel, detected=left_detected)
    line_rt.update_line(right_fit_pixel, detected=right_detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit_pixel[0] * ploty**2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty**2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # Prepare output image
    img_fit = np.dstack((img_birdeye, img_birdeye, img_birdeye))
    window_img = np.zeros_like(img_fit)

    # Color in left and right line pixels
    img_fit[nonzero[0][left_lane_inds], nonzero[1][left_lane_inds]] = [255, 0, 0]
    img_fit[nonzero[0][right_lane_inds], nonzero[1][right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv.fillPoly(window_img, np.array([left_line_pts], np.int_), (0, 255, 0))
    cv.fillPoly(window_img, np.array([right_line_pts], np.int_), (0, 255, 0))
    result = cv.addWeighted(img_fit, 1, window_img, 0.3, 0)

    return result, line_lt, line_rt


def draw_back_onto_the_road(
    img_undistorted: cv.Mat, Minv: cv.Mat, line_lt: Line, line_rt: Line, keep_state: bool
) -> cv.Mat:
    """Draws the detected lane boundaries and fills the lane area.

    1. Dewarp the road and fill the lane area with green color
    2. Draw the detected lane boundaries on the dewarped image (in red and blue color)

    Parameters
    ----------
    img_undistorted : cv.Mat
        The undistorted image the lane boundaries should be drawn on
    Minv : _type_
        The inverse perspective transform matrix
    line_lt : _type_
        The left line object
    line_rt : _type_
        The right line object
    keep_state : _type_
        If True, the average fit is used for drawing the lane boundaries

    Returns
    -------
    cv.Mat
        The image in undistorted perspective with the lane boundaries and the filled lane area
    """
    h, w, _ = img_undistorted.shape

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]  # type: ignore
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]  # type: ignore

    # Draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv.fillPoly(road_warp, np.array([pts], np.int_), (0, 255, 0))
    road_dewarped = cv.warpPerspective(road_warp, Minv, (w, h))  # Warp back to original image space

    blend_onto_road = cv.addWeighted(img_undistorted, 1.0, road_dewarped, 0.3, 0)

    # Separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
    line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
    line_dewarped = cv.warpPerspective(line_warp, Minv, (w, h))

    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    blend_onto_road = cv.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.0)

    return blend_onto_road


def calculate_mean_squared_error(last_fit: cv.Mat, new_fit: cv.Mat, height: int) -> float:
    """Calculates the mean squared error of the `new_fit` polynomial compared to given `last_fit` polynomial.

    Parameters
    ----------
    last_fit : cv.Mat
        Polynomial of 2 degrees
    new_fit : cv.Mat
        Polynomial of 2 degrees
    height : int
        Max y value the error should be calculated of

    Returns
    -------
    float
        Mean Squared error
    """
    y_values = np.linspace(0, height - 1, height).astype(np.int32)

    squared_error = np.sum(
        (
            ((last_fit[0] * y_values**2) + (last_fit[1] * y_values) + last_fit[2])
            - ((new_fit[0] * y_values**2) + (new_fit[1] * y_values) + new_fit[2])
        )
        ** 2
    ) / len(y_values)

    return float(squared_error)

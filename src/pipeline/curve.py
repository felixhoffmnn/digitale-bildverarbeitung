import cv2 as cv
import numpy as np
from loguru import logger
from numpy.linalg import LinAlgError


def lane_sliding_windows(img, line, nonzero, n_windows, window_height, margin, h, x_current, minpix=25):
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    nonzero_y, nonzero_x = nonzero

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
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
    line.all_x, line.all_y = nonzero_x[lane_inds], nonzero_y[lane_inds]

    detected = True
    if not list(line.all_x) or not list(line.all_y):
        fit_pixel = line.last_fit_pixel
        detected = False
    else:
        fit_pixel = np.polyfit(line.all_y, line.all_x, 2)

    return lane_inds, fit_pixel, detected


def get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9):
    """
    Get polynomial coefficients for lane-lines detected in an binary image.
    :param img_birdeye: input bird's eye view binary image
    :param line_lt: left lane-line previously detected
    :param line_rt: left lane-line previously detected
    :param n_windows: number of sliding windows used to search for the lines
    :param verbose: if True, display intermediate output
    :return: updated lane lines and output image
    """
    h, w = img_birdeye.shape

    off_x = w * 0.04

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_birdeye[h // 2 : -15, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img_birdeye, img_birdeye, img_birdeye)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    midpoint = len(histogram) // 2

    leftx_base = np.argmax(histogram[: int(midpoint - off_x)])
    rightx_base = np.argmax(histogram[int(midpoint + off_x) :]) + midpoint

    # Set height of windows
    window_height = int(h / n_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_birdeye.nonzero()
    nonzero_y, nonzero_x = nonzero

    margin = 50  # width of the windows +/- margin

    left_lane_inds, left_fit_pixel, left_detected = lane_sliding_windows(
        img_birdeye, line_lt, nonzero, n_windows, window_height, margin, h, leftx_base
    )
    right_lane_inds, right_fit_pixel, right_detected = lane_sliding_windows(
        img_birdeye, line_rt, nonzero, n_windows, window_height, margin, h, rightx_base
    )

    line_lt.update_line(left_fit_pixel, detected=left_detected)
    line_rt.update_line(right_fit_pixel, detected=right_detected)

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    return out_img, line_lt, line_rt


def get_fit_by_previous_line(name, line, nonzero, margin, h):
    nonzero_y, nonzero_x = nonzero

    fit_pixel = line.last_fit_pixel

    # Identify the x and y positions of all nonzero pixels in the image
    lane_inds = (nonzero_x > (fit_pixel[0] * (nonzero_y**2) + fit_pixel[1] * nonzero_y + fit_pixel[2] - margin)) & (
        nonzero_x < (line.last_fit_pixel[0] * (nonzero_y**2) + fit_pixel[1] * nonzero_y + fit_pixel[2] + margin)
    )

    # Extract left and right line pixel positions
    line.all_x, line.all_y = nonzero_x[lane_inds], nonzero_y[lane_inds]

    # If there are no pixels found, do not update the line
    # Otherwise, calculate the new fit
    detected = True
    if not list(line.all_x) or not list(line.all_y):
        fit_pixel = line.last_fit_pixel
        detected = False
    else:
        try:
            fit_pixel = np.polyfit(line.all_y, line.all_x, 2)
        except LinAlgError:
            fit_pixel = line.last_fit_pixel
            detected = False

    mean_squared_error = calculate_mean_squared_error(fit_pixel, line.last_fit_pixel, h)

    if mean_squared_error > 205:
        logger.debug(f"{name}_mean_squared_error: {mean_squared_error}")
        fit_pixel = line.last_fit_pixel
        line.detected = False
        line.error_count += 1

    return lane_inds, fit_pixel, detected


def get_fits_by_previous_fits(img_birdeye, line_lt, line_rt):
    """
    Get polynomial coefficients for lane-lines detected in an binary image.
    This function starts from previously detected lane-lines to speed-up the search of lane-lines in the current frame.
    :param img_birdeye: input bird's eye view binary image
    :param line_lt: left lane-line previously detected
    :param line_rt: left lane-line previously detected
    :param verbose: if True, display intermediate output
    :return: updated lane lines and output image
    """
    h, w = img_birdeye.shape

    # Gets the last polynomial coefficients
    left_fit_pixel = line_lt.last_fit_pixel
    right_fit_pixel = line_rt.last_fit_pixel

    # logger.debug(f"left_fit_pixel_old: {left_fit_pixel_old}")
    # logger.debug(f"right_fit_pixel: {right_fit_pixel}")

    # Set the width of the windows +/- margin
    nonzero = img_birdeye.nonzero()
    margin = 50
    n_windows = 9
    window_height = int(h / n_windows)

    histogram = np.sum(img_birdeye[h // 2 : -15, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    midpoint = len(histogram) // 2

    off_x = w * 0.04

    leftx_base = np.argmax(histogram[: int(midpoint - off_x)])
    rightx_base = np.argmax(histogram[int(midpoint + off_x) :]) + midpoint

    sum_light = np.sum(img_birdeye)

    if line_lt.error_count > 30 and sum_light > 200000:
        logger.info(f"Refitting left lane: error count: {line_lt.error_count}")
        left_lane_inds, left_fit_pixel, left_detected = lane_sliding_windows(
            img_birdeye, line_lt, nonzero, 9, 80, margin, h, leftx_base
        )
    else:
        left_lane_inds, left_fit_pixel, left_detected = get_fit_by_previous_line("left", line_lt, nonzero, margin, h)

    if line_rt.error_count > 30 and sum_light > 200000:
        logger.info(f"Refitting right lane: error count: {line_rt.error_count}")
        right_lane_inds, right_fit_pixel, right_detected = lane_sliding_windows(
            img_birdeye, line_rt, nonzero, n_windows, window_height, margin, h, rightx_base
        )
    else:
        right_lane_inds, right_fit_pixel, right_detected = get_fit_by_previous_line(
            "right", line_rt, nonzero, margin, h
        )

    line_lt.update_line(left_fit_pixel, detected=left_detected)
    line_rt.update_line(right_fit_pixel, detected=right_detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit_pixel[0] * ploty**2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty**2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # Create an image to draw on and an image to show the selection window
    img_fit = np.dstack((img_birdeye, img_birdeye, img_birdeye)) * 255
    window_img = np.zeros_like(img_fit)

    # Color in left and right line pixels
    img_fit[nonzero[0][left_lane_inds], nonzero[1][left_lane_inds]] = [255, 0, 0]
    img_fit[nonzero[0][right_lane_inds], nonzero[1][right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
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


def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):
    """
    Draw both the drivable lane area and the detected lane-lines onto the original (undistorted) frame.
    :param img_undistorted: original undistorted color frame
    :param Minv: (inverse) perspective transform matrix used to re-project on original frame
    :param line_lt: left lane-line previously detected
    :param line_rt: right lane-line previously detected
    :param keep_state: if True, line state is maintained
    :return: color blend
    """
    h, w, _ = img_undistorted.shape

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv.fillPoly(road_warp, np.array([pts], np.int_), (0, 255, 0))
    road_dewarped = cv.warpPerspective(road_warp, Minv, (w, h))  # Warp back to original image space

    blend_onto_road = cv.addWeighted(img_undistorted, 1.0, road_dewarped, 0.3, 0)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
    line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
    line_dewarped = cv.warpPerspective(line_warp, Minv, (w, h))

    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    blend_onto_road = cv.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.0)

    return blend_onto_road


def calculate_mean_squared_error(last_fit: list[np.float64], new_fit: list[np.float64], height: int) -> float:
    """Calculates the mean squared error of the `new_fit` polynomial compared to given `last_fit` polynomial.

    Parameters
    ----------
    last_fit : list[np.float64]
        Polynomial of 2 degrees
    new_fit : list[np.float64]
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

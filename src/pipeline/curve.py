import cv2 as cv
import numpy as np
from loguru import logger


def find_lane_pixels(img, nwindows=9, margin=100, minpix=50):
    """Find the lanes in a top-down image

    Given a top-down (np.array), find the lane lines.

    Parameters:
        img (numpy.array): numpy array representation of an image
        nwindows (int, optional): the number of sliding windows
        margin (int, optional): width of the windows +/- margin
        minpix (int, optional): minimum number of pixels found to recenter window
    Returns:
        leftx: x-axis value of the left lane
        lefty: y-axis value of the left lane
        rightx: x-axis value of the right lane
        righty: y-axis value of the right lane
        radius: mean radius in meters of the left and right lane curvature
        offset: distance in meters from the center of the lane
        out_img: numpy array representation of the lane detection image
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ym_per_pixel = 30 / 720
    xm_per_pixel = 3.7 / 700

    left_fit_m = np.polyfit(lefty * ym_per_pixel, leftx * xm_per_pixel, 2)
    right_fit_m = np.polyfit(righty * ym_per_pixel, rightx * xm_per_pixel, 2)

    radius, offset = get_radius_and_offset(left_fit_m, right_fit_m, ym_per_pixel, xm_per_pixel)

    # Highlight lane pixels and draw fit polynomials
    lane_pixel_img = np.dstack((img, img, img)) * 255
    lane_pixel_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    lane_pixel_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, radius, offset, out_img


def draw_polynomial(img, fit):
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x = fit[0] * y**2 + fit[1] * y + fit[2]
    pts = np.array([np.transpose(np.vstack([x, y]))])
    cv.polylines(img, np.int_(pts), isClosed=False, color=(255, 255, 0), thickness=5)


def fit_polynomial(img):
    """Extend the lane lines

    Given an image (np.array), fit a second order polynomial to the detected lines.

    Parameters:
        img (numpy.array): numpy array representation of an image
    Returns:
        ploty: np.array of evenly spaced numbers over a specified interval
        left_fit (list): polynomial coefficients of the left lane, highest power first
        right_fit (list): polynomial coefficients of the right lane, highest power first
        radius: mean radius in meters of the left and right lane curvature
        offset: distance in meters from the center of the lane
    """

    # Find our lane pixels first
    leftx, lefty, rightx, righty, radius, offset, out_img = find_lane_pixels(img)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    try:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        logger.error("The function failed to fit a line!")
        left_fitx = 1 * ploty**2 + 1 * ploty
        right_fitx = 1 * ploty**2 + 1 * ploty

    ## Visualization ##
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    draw_polynomial(out_img, left_fit)
    draw_polynomial(out_img, right_fit)

    return ploty, left_fit, right_fit, radius, offset, out_img


def draw_lanes(original_img, img, left_fit, right_fit, Minv):
    """Fill the detected lane

    Given an image (np.array), draw a filled polygon of the lane

    Parameters:
        original_img (numpy.array): numpy array representation of the original image
        img (numpy.array): numpy array representation of an image
        left_fit (list): polynomial coefficients of the left lane, highest power first
        right_fit (list): polynomial coefficients of the right lane, highest power first
        Minv (numpy.array): the inverse perspective transform

    Returns:
        result: numpy array representation of the lane detected image
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    leftx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result


def get_radius_and_offset(left_fit, right_fit, ym_per_pixel, xm_per_pixel):
    """Get the radius of a lane and the offset of the camera

    Given the the polynomial fit of two lines and the meters per pixel of an image,
    calculate the mean radius of the curvature of the lane and the camera's distance
    from the center of the lane.

    Parameters:
        left_fit (list): polynomial coefficients of the left lane, highest power first
        right_fit (list): polynomial coefficients of the right lane, highest power first
        ym_per_pixel (float):
        xm_per_pixel (float):

    Returns:
        radius (float): median radius in meters of the left and right lane curvature
        offset (float) : distance in meters from the center of the lane
    """
    left_curverad = ((1 + (2 * left_fit[0] * 720 * ym_per_pixel + left_fit[1]) ** 2) ** (3 / 2)) / np.abs(
        2 * left_fit[0]
    )
    right_curverad = ((1 + (2 * right_fit[0] * 720 * ym_per_pixel + right_fit[1]) ** 2) ** (3 / 2)) / np.abs(
        2 * right_fit[0]
    )

    left_lane = left_fit[0] * (720 * ym_per_pixel) ** 2 + left_fit[1] * 720 * ym_per_pixel + left_fit[2]
    right_lane = right_fit[0] * (720 * ym_per_pixel) ** 2 + right_fit[1] * 720 * ym_per_pixel + right_fit[2]

    radius = np.mean([left_curverad, right_curverad])
    offset = [640 * xm_per_pixel - np.mean([left_lane, right_lane]), right_lane - left_lane]
    return radius, offset


def add_calculations(result_img, radius, offset):
    """Add the radius and offset to an image

    Parameters:
        result_img (numpy.array): numpy array representation of an image
        radius (float): median radius in meters of the left and right lane curvature
        offset (float) : distance in meters from the center of the lane

    Returns:
        result_img (numpy.array): numpy array representation of an image with added text
    """
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    cv.putText(
        result_img,
        f"Radius of Curvature: {round(radius)} m",
        (50, 100),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        4,
    )
    cv.putText(
        result_img,
        f"Offset: {round(offset[0], 3)} m",
        (50, 200),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        4,
    )

    return result_img

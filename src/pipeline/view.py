import cv2 as cv
import numpy as np
from loguru import logger


def overlay_frames(undist: cv.Mat, thresh: cv.Mat, transform: cv.Mat, img_fit: cv.Mat):
    """Function which combines multiple images into one pretty output image

    !!! note "Source"
        - https://github.com/sidroopdaska/SelfDrivingCar/blob/master/AdvancedLaneLinesDetection/lane_tracker.ipynb

    Parameters
    ----------
    undist : cv.Mat
        Undistorted image blended with detected lane lines
    thresh : cv.Mat
        The thresholded binary image
    transform : cv.Mat
        The bird's eye view of the thresholded image
    img_fit : cv.Mat
        The bird's eye view with detected lane lines highlighted

    Returns
    -------
    cv.Mat
        The composed image containing
    """
    # Set the height and width, as well as offsets in pixels
    h, w = undist.shape[:2]
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)
    off_x, off_y = 20, 15

    # Define a gray mask for separating the additional images from the original image
    mask = undist.copy()
    mask = cv.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(0, 0, 0), thickness=cv.FILLED)
    undist = cv.addWeighted(src1=mask, alpha=0.2, src2=undist, beta=0.8, gamma=0)

    # Add the top left thumbnail (thresholded binary image)
    thumb_binary = cv.resize(thresh, dsize=(thumb_w, thumb_h))
    thumb_binary = np.stack([thumb_binary] * 3, axis=2)
    undist[off_y : thumb_h + off_y, off_x : off_x + thumb_w, :] = thumb_binary

    # Add the center thumbnail (bird's eye view)
    thumb_birdeye = cv.resize(transform, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.stack([thumb_birdeye] * 3, axis=2)
    undist[off_y : thumb_h + off_y, int((w / 2) - (thumb_w / 2)) : int((w / 2) + (thumb_w / 2)), :] = thumb_birdeye

    # Add the top right thumbnail (bird's eye view with detected lane lines)
    thumb_img_fit = cv.resize(img_fit, dsize=(thumb_w, thumb_h))
    undist[off_y : thumb_h + off_y, w - off_x - thumb_w : w - off_x, :] = thumb_img_fit

    return undist

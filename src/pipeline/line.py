import sys
from collections import deque
from typing import Any

import cv2 as cv
import numpy as np
from loguru import logger


class Line:
    """
    Class to model a lane-line.
    """

    def __init__(self, buffer_len: int = 10):
        # Flag to mark if the line was detected the last iteration
        self.detected = False
        # Number of consecutive errors (MSE > threshold)
        self.error_count = 0

        # Polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        # List of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = deque(maxlen=buffer_len)  # type: ignore

        # Store all pixels coordinates (x, y) of line detected
        self.all_x = None
        self.all_y = None

    def update_line(self, new_fit_pixel: Any, detected: bool) -> None:
        """Update the given line object with new polynomial coefficients.

        Parameters
        ----------
        new_fit_pixel : Any
            The new polynomial coefficients.
        detected : bool
            Flag to mark if the line was detected the last iteration.
        """
        self.detected = detected

        self.last_fit_pixel = new_fit_pixel
        self.recent_fits_pixel.append(self.last_fit_pixel)

    def draw(
        self, mask: cv.Mat, color: tuple[int, int, int] = (255, 0, 0), line_width: int = 10, average: bool = False
    ) -> cv.Mat:
        """Draws a line onto the warped image

        Parameters
        ----------
        mask : cv.Mat
            The warped blank image onto which the lane is to be drawn
        color : tuple[int, int, int], optional
            Color of the lanes which are to be drawn, by default (255, 0, 0)
        line_width : int, optional
            Width of the line that is to be drawn, by default 10
        average : bool, optional
            If True, the average fit is used for drawing the lane boundaries, by default False

        Returns
        -------
        cv.Mat
            Returns the lane drawn onto the warped blank image
        """
        h, _, _ = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        if coeffs is None:
            logger.error("Something went wrong, no coefficients found for line.")
            sys.exit(1)

        line_center = coeffs[0] * plot_y**2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Some magic here to recast the x and y points into usable format for cv.fillPoly()
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        # Draw the lane onto the warped blank image
        return cv.fillPoly(mask, [np.int32(pts)], color)

    @property
    # Average of polynomial coefficients of the last N iterations
    def average_fit(self) -> np.ndarray[Any, Any]:
        return np.mean(self.recent_fits_pixel, axis=0)  # type: ignore

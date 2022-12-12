import sys

import cv2 as cv
import numpy as np
from loguru import logger


def get_calib_params(calib_images: list[str], width: int = 9, height: int = 6) -> tuple[cv.Mat, cv.Mat]:
    """Calculate the camera calibration parameters based on the calibration images.

    Parameters
    ----------
    calib_images : list[str]
        The list of paths to the calibration images.
    width : int, optional
        The number of inner corners in the calibration images in the x direction. _By default `9`._
    height : int, optional
        The number of inner corners in the calibration images in the y direction. _By default `6`._

    Returns
    -------
    tuple[cv.Mat, cv.Mat]
        The camera matrix and the distortion coefficients.
    """
    logger.info("Calibrating camera...")

    img_gray = None

    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[:width, :height].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for filename in calib_images:
        img_gray = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(img_gray, (width, height), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if img_gray is None:
        logger.error("Something went wrong. Exiting...")
        sys.exit(1)

    _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

    logger.success("Camera calibrated!")
    return mtx, dist

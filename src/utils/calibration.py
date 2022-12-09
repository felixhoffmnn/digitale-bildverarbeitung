import cv2 as cv
import numpy as np


def get_calibration_params(calib_images: list[str], width: int, height: int):
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

    _, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

    return mtx, dist

import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Add docstring
def transform_perspective(img_gray: cv.Mat) -> tuple[cv.Mat, cv.Mat]:
    img_size = (img_gray.shape[1], img_gray.shape[0])
    offset = 300

    src = np.array([(701, 459), (1055, 680), (265, 680), (580, 459)], dtype=np.float32)
    dst = np.array(
        [
            (img_size[0] - offset, 0),
            (img_size[0] - offset, img_size[1]),
            (img_size[0] - img_size[0] + offset, img_size[1]),
            (img_size[0] - img_size[0] + offset, 0),
        ],
        dtype=np.float32,
    )

    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)

    transformed_img = cv.warpPerspective(img_gray, M, img_size, flags=cv.INTER_LINEAR)

    return transformed_img, Minv


# TODO: Add docstring
def undist_img(img: cv.Mat, mtx: cv.Mat, dist: cv.Mat) -> cv.Mat:
    img_dst = cv.undistort(img, mtx, dist)
    return img_dst

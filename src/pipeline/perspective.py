import cv2 as cv
import numpy as np
from loguru import logger


# TODO: Add docstring
def undist_img(img: cv.Mat, ca_param: tuple[cv.Mat, cv.Mat]) -> cv.Mat:
    img_dst = cv.undistort(img, ca_param[0], ca_param[1])
    return img_dst


# TODO: Refactor this function
# TODO: Add docstring
def transform_perspective(
    img_gray: cv.Mat,
) -> tuple[cv.Mat, cv.Mat]:  # , destination: cv.Mat
    """Transforms the region of interest into a bird's eye view.

    Parameters
    ----------
    img_gray : cv.Mat
        The region of interest as a grayscale image.

    Returns
    -------
    tuple[cv.Mat, cv.Mat]
        The transformed image and the inverse transformation matrix.
    """
    h, w = img_gray.shape[:2]
    offset = 300

    # TODO: Outsource this because it is the same calculation every time
    src = np.array([(701, 459), (1055, 680), (265, 680), (580, 459)], dtype=np.float32)
    dst = np.array(
        [
            (w - offset, 0),
            (w - offset, h),
            (offset, h),
            (offset, 0),
        ],
        dtype=np.float32,
    )

    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)

    transformed_img = cv.warpPerspective(img_gray, M, (w, h), flags=cv.INTER_LINEAR)

    return transformed_img, Minv

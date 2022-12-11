import cv2 as cv
import numpy as np
from loguru import logger


# TODO: Add docstring
def undist_img(img: cv.Mat, ca_param: tuple[cv.Mat, cv.Mat]) -> cv.Mat:
    img_dst = cv.undistort(img, ca_param[0], ca_param[1])
    return img_dst


# TODO: Refactor this function
# TODO: Add docstring
def region_of_interest(img_gray: cv.Mat, kitti: bool) -> tuple[cv.Mat, cv.Mat]:
    h, w = img_gray.shape[:2]

    # Find region of interest
    if kitti:
        vertices = np.array(
            [
                (w * 0.75, h * 0.52),  # Top-right corner
                (w, h),  # Bottom-right corner
                (0, h),  # Bottom-left corner
                (w * 0.39, h * 0.52),  # Top-left corner
            ],
            dtype=np.float32,
        )
    else:
        vertices = np.array(
            [
                (w * 0.6, h * 0.65),  # Top-right corner
                (w * 0.9, h * 0.93),  # Bottom-right corner
                (w * 0.1, h * 0.93),  # Bottom-left corner
                (w * 0.4, h * 0.65),  # Top-left corner
            ],
            np.float32,
        )

    # defining a blank mask to start with
    mask = np.zeros_like(img_gray)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv.fillPoly(mask, np.array([vertices], np.int32), [255])

    # returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img_gray, mask)

    return masked_image, vertices


# TODO: Refactor this function
# TODO: Add docstring
def transform_perspective(img_gray: cv.Mat, vertices: cv.Mat) -> tuple[cv.Mat, cv.Mat]:  # , destination: cv.Mat
    """Transforms the region of interest into a bird's eye view.

    Parameters
    ----------
    img_gray : cv.Mat
        The region of interest as a grayscale image.
    vertices : cv.Mat
        The vertices of the region of interest.

    Returns
    -------
    tuple[cv.Mat, cv.Mat]
        The transformed image and the inverse transformation matrix.
    """
    h, w = img_gray.shape[:2]
    offset = w * 0.235

    dst = np.array(
        [
            [w - offset, 0],  # Top-right corner
            [w - offset, h],  # Bottom-right corner
            [offset, h],  # Bottom-left corner
            [offset, 0],  # Top-left corner
        ],
        dtype=np.float32,
    )

    M = cv.getPerspectiveTransform(vertices, dst)
    Minv = cv.getPerspectiveTransform(dst, vertices)

    transformed_img = cv.warpPerspective(img_gray, M, (w, h), flags=cv.INTER_LINEAR)

    return transformed_img, Minv

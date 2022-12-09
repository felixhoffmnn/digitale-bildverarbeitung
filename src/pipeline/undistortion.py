import cv2 as cv


# TODO: Add docstring
def undist_img(img: cv.Mat, mtx: cv.Mat, dist: cv.Mat) -> cv.Mat:
    img_dst = cv.undistort(img, mtx, dist)
    return img_dst

import sys
from glob import glob

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from loguru import logger

from src.pipeline.curve_fitting import fit_polynomial
from src.pipeline.perspective_transformation import transform_perspective
from src.pipeline.segmentation import segment_img
from src.pipeline.thresholding import thresh_img
from src.pipeline.undistortion import undist_img
from src.utils.shell import clear_shell, get_int, print_options

# from utils.calibration import get_calibration_params


def pipeline(img: cv.Mat, mtx: cv.Mat, dist: cv.Mat) -> cv.Mat:
    # Undistortion("calibration", mtx, dist),
    # Segmentation("segmentation"),
    # PerspectiveTransformation("perspective_transformation"),
    # Thresholding("thresholding"),
    # CurveFitting("curve_fitting"),

    undist = undist_img(img, mtx, dist)
    # TODO: Think about blurring image before thresholding
    thresh = thresh_img(undist)
    region = segment_img(thresh)
    transform = transform_perspective(region)
    poly = fit_polynomial(transform)

    return poly


def main() -> None:
    clear_shell()

    # Calibrate camera based on chessboard images
    # calib_images = glob("./data/exam/calib/*.jpg")
    # logger.debug(f"Found {len(calib_images)} calibration images.")
    # mtx, dist = get_calibration_params(calib_images, 9, 6)
    mtx = cv.Mat(
        np.array(
            [
                [1.15777930e03, 0.00000000e00, 6.67111054e02],
                [0.00000000e00, 1.15282291e03, 3.86128937e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
    )
    dist = cv.Mat(np.array([[-0.24688775, -0.02373133, -0.00109842, 0.00035108, -0.00258571]]))

    options_1: list[str] = ["images", "calib", "videos"]
    print_options(options_1)
    user_input_1 = get_int(options_1)

    if user_input_1 is None:
        logger.error("Invalid input. Exiting...")
        sys.exit(1)

    glob_files = glob(f"./data/exam/{options_1[user_input_1 - 1]}/*")

    # The user wants to run the pipeline on images
    if user_input_1 == 1 or user_input_1 == 2:
        logger.debug("Running pipeline on images...")
        print_options(glob_files)
        user_input_2 = get_int(glob_files)

        if user_input_2 is None:
            logger.error("Invalid input. Exiting...")
            sys.exit(1)

        # Load the image and run the pipeline
        img = cv.cvtColor(cv.imread(glob_files[user_input_2 - 1]), cv.COLOR_BGR2RGB)
        processed_img = pipeline(img, mtx, dist)

        plt.imshow(processed_img)
        plt.show()

    # The user wants to run the pipeline on videos
    if user_input_1 == 3:
        logger.debug("Running pipeline on videos...")
        print_options(glob_files)
        user_input_2 = get_int(glob_files)

        if user_input_2 is None:
            logger.error("Invalid input. Exiting...")
            sys.exit(1)

        video_path = glob_files[user_input_2 - 1]
        cap = cv.VideoCapture(video_path)

        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("The video has ended. Exiting...")
                break

            # Apply pipeline
            processed_frame = cv.cvtColor(pipeline(frame, mtx, dist), cv.COLOR_RGB2BGR)

            # render new frame
            cv.imshow("frame", processed_frame)
            if cv.waitKey(1) == ord("q"):
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)

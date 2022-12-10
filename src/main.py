import os
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from turtle import right

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2 import dft, threshold
from fire import Fire
from loguru import logger

from src.pipeline.curve import draw_lanes, fit_polynomial
from src.pipeline.perspective import transform_perspective, undist_img
from src.pipeline.region import region_of_interest
from src.pipeline.thresholding import get_line_markings, thresh_img
from src.utils.shell import clear_shell, get_int, print_options

# from utils.calibration import get_calibration_params


@dataclass
class ConvertedImage:
    undist: cv.Mat | None = None
    gaussian: cv.Mat | None = None
    thresh: cv.Mat | None = None
    region: cv.Mat | None = None
    transform: cv.Mat | None = None
    poly: cv.Mat | None = None
    draw: cv.Mat | None = None


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    rigt_line = make_coordinates(image, right_fit_average)
    left_line = make_coordinates(image, left_fit_average)

    return np.array([left_line, rigt_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image


def gaussian_blur(img: cv.Mat, kernel_size: int = 5) -> cv.Mat:
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)


def pipeline(img: cv.Mat, mtx: cv.Mat, dist: cv.Mat) -> ConvertedImage:
    # Undistortion("calibration", mtx, dist),
    # Segmentation("segmentation"),
    # PerspectiveTransformation("perspective_transformation"),
    # Thresholding("thresholding"),
    # CurveFitting("curve_fitting"),

    normalize = cv.normalize(img, img, 15, 225, cv.NORM_MINMAX)
    undist = undist_img(normalize, mtx, dist)
    gaussian = gaussian_blur(undist)
    # TODO: Think about blurring image before thresholding
    thresh = thresh_img(gaussian)
    region = region_of_interest(thresh)
    # transform, Minv = transform_perspective(region)
    # ploty, left_fit, right_fit, radius, offset, polyfit_img = fit_polynomial(transform)
    # draw = draw_lanes(img, transform, left_fit, right_fit, Minv)

    return ConvertedImage(undist, gaussian, thresh, region)  # ,transform, polyfit_img, draw)


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

    options_1: list[str] = os.listdir("./data/exam")
    print_options(options_1)
    user_input_1 = get_int(options_1)

    if user_input_1 is None:
        logger.error("Invalid input. Exiting...")
        sys.exit(1)

    glob_files = glob(f"./data/exam/{options_1[user_input_1 - 1]}/*")

    # The user wants to run the pipeline on images
    if options_1[user_input_1 - 1] == "images" or options_1[user_input_1 - 1] == "calib":
        logger.debug("Running pipeline on images...")
        print_options(glob_files)
        user_input_2 = get_int(glob_files)

        if user_input_2 is None:
            logger.error("Invalid input. Exiting...")
            sys.exit(1)

        # Load the image and run the pipeline
        img = cv.cvtColor(cv.imread(glob_files[user_input_2 - 1]), cv.COLOR_BGR2RGB)
        converted_image = pipeline(img, mtx, dist)

        img_to_plot = converted_image.region
        if img_to_plot is None:
            logger.error("The image could not be converted. Exiting...")
            sys.exit(1)

        plt.imshow(img_to_plot)
        plt.show()

    # The user wants to run the pipeline on videos
    if options_1[user_input_1 - 1] == "videos":
        logger.debug("Running pipeline on videos...")
        print_options(glob_files)
        user_input_2 = get_int(glob_files)

        if user_input_2 is None:
            logger.error("Invalid input. Exiting...")
            sys.exit(1)

        video_path = glob_files[user_input_2 - 1]
        cap = cv.VideoCapture(video_path)

        frame_count = 0
        start_time = datetime.now()

        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("The video has ended. Exiting...")
                break

            # Apply pipeline
            converted_frame = pipeline(frame, mtx, dist)

            frame_to_plot = converted_frame.region
            if frame_to_plot is None or converted_frame.undist is None:
                logger.error("The image could not be converted. Exiting...")
                sys.exit(1)

            lines = cv.HoughLinesP(frame_to_plot, 1, np.pi / 180, 100, minLineLength=60, maxLineGap=5)
            averaged_lines = average_slope_intercept(frame_to_plot, lines)
            lines_image = display_lines(frame_to_plot, averaged_lines)
            combo_image = cv.addWeighted(cv.cvtColor(converted_frame.undist, cv.COLOR_RGB2GRAY), 0.8, lines_image, 1, 1)

            # render new frame
            cv.imshow("frame", combo_image)
            if cv.waitKey(1) == ord("q"):
                break

            frame_count += 1
            if frame_count % 100 == 0:
                frame_rate = frame_count / (datetime.now() - start_time).total_seconds()
                logger.info(f"Frames per second: {frame_rate:.2f}")

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)

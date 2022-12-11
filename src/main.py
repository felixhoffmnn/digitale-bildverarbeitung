import os
import sys
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from turtle import right

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from loguru import logger

from src.pipeline.curve import (
    draw_back_onto_the_road,
    get_fits_by_previous_fits,
    get_fits_by_sliding_windows,
)
from src.pipeline.line import Line
from src.pipeline.perspective import transform_perspective, undist_img
from src.pipeline.region import region_of_interest
from src.pipeline.threshold import thresh_img
from src.utils.shell import clear_shell, get_int, print_options

processed_frames = 0  # counter of frames processed (when processing video)
line_lt = Line(buffer_len=10)  # line on the left of the lane
line_rt = Line(buffer_len=10)  # line on the right of the lane


def apply_blur(img: cv.Mat, kernel_size: int = 5) -> cv.Mat:
    gausian = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
    median = cv.medianBlur(gausian, kernel_size)
    return median


def pipeline(img: cv.Mat, mtx: cv.Mat, dist: cv.Mat, keep_state: bool = True) -> list[cv.Mat]:  # -> ConvertedImage:
    global line_lt, line_rt, processed_frames

    undistort = undist_img(img, mtx, dist)
    gaussian = apply_blur(undistort)
    line = thresh_img(gaussian)
    # binar = binarize(gaussian)
    region = region_of_interest(line)
    transform, Minv = transform_perspective(region)
    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(transform, line_lt, line_rt)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(transform, line_lt, line_rt, n_windows=9)
    blend_on_road = draw_back_onto_the_road(undistort, Minv, line_lt, line_rt, keep_state)

    return [undistort, gaussian, line, region, transform, img_fit, blend_on_road]  # , region, transform, poly, draw


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
    # options_1 = [x for x in options_1 if x != "calib"]
    print_options(options_1)
    user_input_1 = get_int(options_1)

    if user_input_1 is None:
        logger.error("Invalid input. Exiting...")
        sys.exit(1)

    glob_files = glob(f"./data/exam/{options_1[user_input_1 - 1]}/*")

    # The user wants to run the pipeline on images
    if (
        options_1[user_input_1 - 1] == "images"
        or options_1[user_input_1 - 1] == "calib"
        or options_1[user_input_1 - 1] == "optimize"
        or options_1[user_input_1 - 1] == "kitti"
    ):
        logger.debug("Running pipeline on images...")
        print_options(glob_files)
        user_input_2 = get_int(glob_files)

        if user_input_2 is None:
            logger.error("Invalid input. Exiting...")
            sys.exit(1)

        # Load the image and run the pipeline
        img = cv.cvtColor(cv.imread(glob_files[user_input_2 - 1]), cv.COLOR_BGR2RGB)
        converted_image = pipeline(img, mtx, dist, keep_state=False)

        img_to_plot = converted_image[-1]
        if img_to_plot is None:
            logger.error("The image could not be converted. Exiting...")
            sys.exit(1)

        # img_to_plot = cv.cvtColor(img_to_plot, cv.COLOR_GRAY2RGB)

        # plt.plot(hist(img_to_plot))
        plt.imshow(img_to_plot)
        plt.show()

    # The user wants to run the pipeline on videos
    if options_1[user_input_1 - 1] == "videos":
        logger.debug("Running pipeline on videos...")
        filtered_paths = [path for path in glob_files if path != "./data/exam/videos/harder_challenge_video.mp4"]
        print_options(filtered_paths)
        user_input_2 = get_int(filtered_paths)

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
            # converted_frame = create_view(converted_frame)

            frame_to_plot = converted_frame[-1]
            if frame_to_plot is None:
                logger.error("The image could not be converted. Exiting...")
                sys.exit(1)

            # render new frame
            cv.imshow("frame", frame_to_plot)
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

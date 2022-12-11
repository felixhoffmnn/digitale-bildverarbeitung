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
from src.pipeline.perspective import (
    region_of_interest,
    transform_perspective,
    undist_img,
)
from src.pipeline.threshold import thresh_img
from src.pipeline.view import overlay_frames
from src.utils.calibration import get_calibration_params
from src.utils.shell import clear_shell, get_int, print_options

processed_frames = 0  # counter of frames processed (when processing video)
line_lt = Line(buffer_len=10)  # line on the left of the lane
line_rt = Line(buffer_len=10)  # line on the right of the lane
Minv = None
img_thresh = None
img_region = None
img_warped = None
img_poly = None


def apply_blur(img: cv.Mat, kernel_size: int = 3) -> cv.Mat:
    gausian = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
    median = cv.medianBlur(gausian, kernel_size)
    return median


def pipeline(
    img_rgb: cv.Mat, ca_param: tuple[cv.Mat, cv.Mat], pretty: bool, kitti: bool, resize: bool, keep_state: bool = True
) -> list[cv.Mat]:
    """Pipeline to process an image.

    1. `undistort`: Applyes the camera calibration matrix and distortion coefficients to a raw image.
    2. `gaussian`: Blur the image with a Gaussian filter.
    3. `thresh`: Applies a threshold to the image (Sobel, HLS, HSV, Gray).
    4. `region`: Polyfill only the region of interest.
    5. `transform`: Transform the image to a bird's eye view.
    6. `poly`: Find the lane-line pixels based on the previous frame.
    7. `draw`: Draw the lane back onto the original image.

    Parameters
    ----------
    img_rgb : cv.Mat
        The image to process.
    ca_param : tuple[cv.Mat, cv.Mat]
        The camera calibration matrix and distortion coefficients.
    pretty : bool
        If the process pipeline should be converted into a view with multiple frame, like
    kitti : bool, optional
        If the KITTI dataset is selected. _By default `False`._
    keep_state : bool, optional
        If the lane detection should be based on the previous frame. _By default `True`._

    Returns
    -------
    list[cv.Mat]
        A list of images, each representing a step in the pipeline.
    """
    global line_lt, line_rt, img_poly, processed_frames, img_thresh, img_region, img_warped, img_poly, Minv

    img_rgb = cv.pyrDown(img_rgb)

    if kitti:
        img_gaussian = apply_blur(img_rgb)
    else:
        img_undistort = undist_img(img_rgb, ca_param)
        img_gaussian = apply_blur(img_undistort)

    if processed_frames % 2 == 0 or keep_state is False:
        img_thresh = thresh_img(img_gaussian, kitti)

        img_region, vertices = region_of_interest(img_thresh, kitti)

        img_warped, Minv = transform_perspective(img_region, vertices)

        if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
            img_poly, line_lt, line_rt = get_fits_by_previous_fits(img_warped, line_lt, line_rt)
        else:
            img_poly, line_lt, line_rt = get_fits_by_sliding_windows(img_warped, line_lt, line_rt, n_windows=9)

    if kitti:
        img_draw = draw_back_onto_the_road(img_rgb, Minv, line_lt, line_rt, keep_state)
    else:
        img_draw = draw_back_onto_the_road(img_undistort, Minv, line_lt, line_rt, keep_state)

    processed_frames += 1

    if pretty:
        if kitti:
            view = overlay_frames(img_draw, img_thresh, img_warped, img_poly)
            return [img_gaussian, img_thresh, img_region, img_warped, img_poly, img_warped, view]
        else:
            view = overlay_frames(img_draw, img_thresh, img_warped, img_poly)
            return [img_undistort, img_gaussian, img_thresh, img_region, img_warped, img_poly, img_warped, view]

    if kitti is False:
        return [img_undistort, img_gaussian, img_thresh, img_region, img_warped, img_poly, img_draw]
    else:
        return [img_gaussian, img_thresh, img_region, img_warped, img_poly, img_draw]


def get_calibration():
    # Calibrate camera based on chessboard images
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
    ca_param = (mtx, dist)
    # calib_images = glob("./data/exam/calib/*.jpg")
    # ca_param = get_calibration_params(calib_images, 9, 6)
    logger.success("Camera calibrated!")

    return ca_param


def main(pretty: bool = True, step_to_plot: int = -1, resize: bool = True) -> None:
    clear_shell()

    ca_param = get_calibration()

    options_1: list[str] = os.listdir("./data/exam")
    print_options(options_1)
    user_input_1 = get_int(options_1)

    if user_input_1 is None:
        logger.error("Invalid input. Exiting...")
        sys.exit(1)

    glob_files = glob(f"./data/exam/{options_1[user_input_1 - 1]}/*")

    user_input = options_1[user_input_1 - 1]

    # The user wants to run the pipeline on images
    if user_input == "images" or user_input == "calib" or user_input == "optimize" or user_input == "kitti":
        logger.debug("Running pipeline on images...")
        print_options(glob_files)
        user_input_2 = get_int(glob_files)

        if user_input_2 is None:
            logger.error("Invalid input. Exiting...")
            sys.exit(1)

        # Load the image and run the pipeline
        img = cv.cvtColor(cv.imread(glob_files[user_input_2 - 1]), cv.COLOR_BGR2RGB)
        converted_image = pipeline(
            img,
            ca_param,
            pretty=pretty,
            kitti=(True if user_input == "kitti" else False),
            resize=resize,
            keep_state=False,
        )

        try:
            img = converted_image[step_to_plot]
        except KeyError:
            logger.error("The image could not be converted. Exiting...")
            sys.exit(1)

        if img is None:
            logger.error("The image could not be converted. Exiting...")
            sys.exit(1)

        # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        # plt.plot(hist(img))
        plt.imshow(img)
        plt.show()

    # The user wants to run the pipeline on videos
    if user_input == "videos":
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
            converted_frame = pipeline(frame, ca_param, pretty, kitti=False, resize=resize, keep_state=True)
            # converted_frame = create_view(converted_frame)

            try:
                frame = converted_frame[step_to_plot]
            except KeyError:
                logger.error("The image could not be converted. Exiting...")
                sys.exit(1)

            if frame is None:
                logger.error("The image could not be converted. Exiting...")
                sys.exit(1)

            # render new frame
            cv.imshow("frame", frame)
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

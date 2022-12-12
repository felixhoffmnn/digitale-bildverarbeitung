import os
import sys
from datetime import datetime
from glob import glob

import cv2 as cv
import matplotlib.pyplot as plt
from fire import Fire
from loguru import logger

from src.pipeline.calibration import get_calib_params
from src.pipeline.lane import (
    draw_back_onto_the_road,
    get_fits_by_previous_fits,
    get_fits_by_sliding_windows,
)
from src.pipeline.line import Line
from src.pipeline.overlay import overlay_frames
from src.pipeline.perspective import (
    region_of_interest,
    transform_perspective,
    undist_img,
)
from src.pipeline.threshold import apply_blur, thresh_img
from src.utils.shell import clear_shell, get_int, print_options

processed_frames = 0  # counter of frames processed (when processing video)
line_lt = Line(buffer_len=10)  # line on the left of the lane
line_rt = Line(buffer_len=10)  # line on the right of the lane


def pipeline(
    img_rgb: cv.Mat, ca_param: tuple[cv.Mat, cv.Mat] | None, pretty: bool, keep_state: bool = True
) -> list[cv.Mat]:
    """Pipeline to process an image.

    1. `img_undistort`: Applies the camera calibration matrix and distortion coefficients to a raw image.
    2. `img_gaussian`: Blur the image with a Gaussian filter.
    3. `img_thresh`: Applies a threshold to the image (Sobel, HLS, HSV, Gray).
    4. `img_region`: Polyfill only the region of interest.
    5. `img_birdeye`: Transform the image to a bird's eye view.
    6. `img_poly`: Find the lane-line pixels based on the previous frame.
    7. `img_draw`: Draw the lane back onto the original image.
    8. `img_overlay`: Overlay the original image with the detected lanes, threshold, and bird's eye view.

    Parameters
    ----------
    img_rgb : cv.Mat
        The image to process.
    ca_param : tuple[cv.Mat, cv.Mat] | None
        The camera calibration matrix and distortion coefficients.
        If KITTI is selected, this parameter is `None`.
    pretty : bool
        If the process pipeline should be converted into a view with multiple frame, like
    keep_state : bool, optional
        If the lane detection should be based on the previous frame. _By default `True`._

    Returns
    -------
    list[cv.Mat]
        A list of images, each representing a step in the pipeline.
    """
    global line_lt, line_rt, processed_frames

    img_rgb = cv.pyrDown(img_rgb)

    img_undistort = None

    if ca_param:
        img_undistort = undist_img(img_rgb, ca_param)
        img_gaussian = apply_blur(img_undistort)
    else:
        img_gaussian = apply_blur(img_rgb)

    img_thresh = thresh_img(img_gaussian, kitti=(False if ca_param else True))
    img_region, vertices = region_of_interest(img_thresh, kitti=(False if ca_param else True))
    img_birdeye, Minv = transform_perspective(img_region, vertices)

    if processed_frames > 0 and keep_state:
        img_poly, line_lt, line_rt = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt)
    else:
        img_poly, line_lt, line_rt = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9)

    img_draw = draw_back_onto_the_road(
        (img_undistort if img_undistort is not None else img_rgb), Minv, line_lt, line_rt, keep_state
    )

    processed_frames += 1

    if pretty:
        img_overlay = overlay_frames(img_draw, img_thresh, img_birdeye, img_poly)
        return [
            (img_undistort if img_undistort is not None else img_rgb),
            img_gaussian,
            img_thresh,
            img_region,
            img_birdeye,
            img_poly,
            img_draw,
            img_overlay,
        ]

    return [
        (img_undistort if img_undistort is not None else img_rgb),
        img_gaussian,
        img_thresh,
        img_region,
        img_birdeye,
        img_poly,
        img_draw,
    ]


def main(pretty: bool = True, step_to_plot: int = -1) -> None:
    """Wrapper for the user interaction and the pipeline.

    Parameters
    ----------
    pretty : bool, optional
        If the pipeline should be converted into a view with multiple frame. _By default `True`._
    step_to_plot : int, optional
        Parameter to force-plot a specific step in the pipeline. _By default `-1`._
        With this parameter, it is possible to plot steps like `img_undistort`, `img_gaussian`,
        `img_thresh`, `img_region`, `img_birdeye`, `img_poly`, `img_draw`, and `img_overlay`
    """
    clear_shell()

    ca_param = None

    # Get the user input (images, video, or KITTI)
    option_1: list[str] = os.listdir("./data/exam")
    print_options(option_1)
    user_input_1 = get_int(option_1)

    if user_input_1 is None:
        logger.error("Invalid input. Exiting...")
        sys.exit(1)

    if len(option_1) <= 0:
        logger.error("No data found. Exiting...")
        sys.exit(1)

    folder = option_1[user_input_1 - 1]

    # List all files in the selected folder
    input_path = f"./data/exam/{folder}/"
    option_2 = os.listdir(input_path)

    if folder == "videos":
        option_2 = [path for path in option_2 if path != "harder_challenge_video.mp4"]

    # Get the user input (which image to process)
    print_options(option_2)
    user_input_2 = get_int(option_2)

    if user_input_2 is None:
        logger.error("Invalid input. Exiting...")
        sys.exit(1)

    file = option_2[user_input_2 - 1]
    process_path = f"./data/exam/{folder}/{file}"

    output_path = f"./data/output/{folder}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f"{output_path}/{file}"

    logger.info(f"Processing {process_path}...")
    logger.info(f"Output will be saved to {output_file}...")

    if folder != "kitti":
        # Get the camera calibration parameters
        calib_images = glob("./data/exam/calib/*.jpg")
        ca_param = get_calib_params(calib_images)

    # Runs the pipeline on images
    if folder == "images" or folder == "calib" or folder == "optimize" or folder == "kitti":
        logger.debug("Running pipeline on images...")

        # Load the image and run the pipeline
        img = cv.cvtColor(cv.imread(process_path), cv.COLOR_BGR2RGB)
        converted_image = pipeline(
            img,
            ca_param,
            pretty,
            keep_state=False,
        )

        # Plot the requested step
        try:
            img = converted_image[step_to_plot]
        except KeyError:
            logger.error("Invalid step requested. Exiting...")
            sys.exit(1)

        if img is None:
            logger.error("The requested step is not available. Exiting...")
            sys.exit(1)

        # Save image to output path
        cv.imwrite(output_file, cv.cvtColor(converted_image[6], cv.COLOR_RGB2BGR))

        # Plot the image
        plt.imshow(img)
        plt.show()

    # Runs the pipeline on videos
    if folder == "videos":
        logger.debug("Running pipeline on videos...")

        # Load the video and run the pipeline
        cap = cv.VideoCapture(process_path)

        # Get the video properties and initialize counter
        frame_count = 0
        frame_width = int(cap.get(3) / 2)
        frame_height = int(cap.get(4) / 2)
        start_time = datetime.now()

        logger.debug(f"Video resolution: {frame_width}x{frame_height}")

        # Initialize the video writer
        out = cv.VideoWriter(output_file, cv.VideoWriter_fourcc("m", "p", "4", "v"), 20, (frame_width, frame_height))

        # Loop over the video
        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("The video has ended. Exiting...")
                break

            # Apply the pipeline
            converted_frame = pipeline(
                frame,
                ca_param,
                pretty,
                keep_state=True,
            )

            # Get the requested step to plot
            try:
                frame = converted_frame[step_to_plot]
            except KeyError:
                logger.error("The image could not be converted. Exiting...")
                sys.exit(1)

            if frame is None:
                logger.error("The image could not be converted. Exiting...")
                sys.exit(1)

            # Save the current frame
            out.write(frame)

            # Show the current frame
            cv.imshow("frame", frame)
            if cv.waitKey(1) == ord("q"):
                break

            frame_count += 1
            if frame_count % 100 == 0:
                frame_rate = frame_count / (datetime.now() - start_time).total_seconds()
                logger.info(f"Frames per second: {frame_rate:.2f}")

        cap.release()
        out.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)

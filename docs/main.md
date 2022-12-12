# Wrapper and Pipeline

The wrapper combines multiple steps required to perform the lane detection. To achieve that, it performs the following steps:

1. Get `option_1` from the user. Option 1 represents the available folder (e.g., `videos`, `images`, `kitti`).
2. Get `option_2` from the user. Option 2 represents the available files (e.g., `video_1.mp4`, `image_1.png`, `kitti_1.png`).
3. If the user selected the **Udacity** dataset, the camera calibration is performed. Otherwise, the calibration is skipped.
4. Preforms the pipeline on either a **video** or **image**.
    1. First, read the image/video.
    2. Convert the color space if necessary.
    3. Applies the pipeline.
    4. Saves the image/video.
    5. Shows the image/video to the user.

::: src.main

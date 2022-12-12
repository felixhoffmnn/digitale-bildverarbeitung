<h1 align="center">
  Computer Vision
</h1>

<h4 align="center">
  Exercises during the 5th semester and a lane detection project based on a given dataset and the KITTI dataset
</h4>

<div align="center">
  <a href="https://github.com/felixhoffmnn/digitale-bildverarbeitung">
    <img src="https://img.shields.io/github/license/felixhoffmnn/digitale-bildverarbeitung"
      alt="License: MIT" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg"
      alt="Python 3.10" />
  </a>
  <a href="https://github.com/psf/black">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  </a>
  <a href="https://github.com/prettier/prettier">
    <img src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat&logo=appveyor"
      alt="Codestyle: Prettier" />
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"
      alt="pre-commit" />
  </a>
</div>

<div align="center">
    <a href="https://github.com/felixhoffmnn/digitale-bildverarbeitung">GitHub</a>
    ·
    <a href="https://digitale-bildverarbeitung.readthedocs.io/en/latest/">Documentation</a>
</div>
<br>

This repository contains exercises and the final project for the Computer Vision course at the DHBW Stuttgart. The lane detection project is based on a given dataset and the KITTI dataset.

## :rocket: Requirements & Usage

> :arrow_up: Python 3.10 is required

1. Install [Poetry](https://python-poetry.org/docs/#installation)
    - Poetry is a dependency manager for Python used in this project
    - (Optional) Setup poetry to use the local `.venv` folder by running `poetry config virtualenvs.in-project true`
2. Run `poetry install` to install all dependencies - Afterwards, run `poetry shell` to activate the virtual environment
3. Install the pre-commit hooks with `poetry run pre-commit install`

> :warning: **Note:** If you are using _Poetry_ it is recommended to use `poetry run <command>` to run commands. This ensures that the `.env` file is loaded and the virtual environment is activated.

<br>

After the setup is complete, use the **following commands** run the lane detection project. Note that you need to be in the **root directory** of the project.

```bash
# If want to run the lane detection with the default settings
poetry run python src/main.py

# With this you can disable the overlay (`pretty` visualization)
poetry run python src/main.py False

# Select a specific step in the pipeline to be shown
# Since `pretty` will not be shown we can disable it
poetry run python src/main.py False 5
```

## :mag: Scope of the Project

**Minimum Requirements**:

-   [x] Camera Calibration (`src/pipeline/calibration.py`)
-   [x] Segmentation of the image/frame (`src/pipeline/perspective.py`)
-   [x] Color thresholding and masking using histogram (`src/pipeline/threshold.py`)
-   [x] Providing minimum `20` fps
-   [x] Increasing the performance by using the previous detected lines and a histogram for fitting the sliding windows (`src/pipeline/lane.py` and `src/pipeline/line.py`)
-   [x] Curve and polynomial fitting (`src/pipeline/lane.py`)
-   [x] Contiguous lane detection for `project_video.mp4`

**Additional Features**:

-   [x] Contiguous lane detection for `challenge_video.mp4`
-   [x] Detecting lines for the KITTI dataset
    -   We don't apply the camera calibration, because KITTI uses a different camera
    -   The angel and view of the camera is different therefore we use a different conversion matrix
    -   To detect some specific lines we use additional color thresholding and sobel filter
-   [x] Thresholding the maximum change of the lines between two frames (`src/pipeline/lane.py`)
    -   If the thesholding is exceeded, the last detected lines are used
    -   If the detected lane is too often detected as a error it will reposition using the sliding windows

## :thought_balloon: Questions

### Approach

As a first step, we had to decide which approach for lane detection we would use. For this, we decided to mainly use the methods for lane detection which were presented to us during lectures. This was done because we knew more about these approaches than about other possible approaches like using neural networks.

After deciding on our approach, we tried to conceptualize the pipeline of functions which we wanted to use to detect the lanes. This first pipeline was then used as a starting point to develop the different functions which were used to detect lanes, although of course changes to the pipeline had to be made during development. Additionally, we also implemented some functions which were not part of the lecture, e.g. sliding windows for line detection, as they were helpful to achieve better results on the given images and videos.

### Alternatives

The big other possible approach would have been using neural networks. Using them, it might have been possible to develop a lane detection which would be more generally applicable. But as they were not a big part of the lecture and as they can be quite tricky to debug if they don't detect what they are supposed to detect, we decided ultimately against using them.

During the development of the lane detection, it was also often necessary to decide which specific functions to use and which not to use. For example, canny edge was not used as the implemented sobel filter was more effective. Furthermore, some functions could not be implemented as they would have worsened the performance significantly.

### Problems and possible Solutions

One problem we encountered was that the lane detection did not work properly on the challenge video. This was because the challenge video had a lot of shadows and the thresholding did not work properly on the shadows. To solve this problem, we developed an algorithm that measures the divergence of a detected line compared to the previous one and only accepts the new line if the divergence is below a certain threshold. More information can be found in the documentation about `lane.py`

The core of our solution for the lane detection are thresholding (which is done in `threshold.py`) and detecting the actual lines with a polyfit operation and sliding windows technique (which is done in `lane.py`). More information, including a detailed description, about these steps in our pipeline can be found in the respective pages in the documentation.

### Learnings

Both thresholding and preprocessing of images are important parts of the lane detection as a good binary image is essential for proper lane detection.

It is quite easy to run into performance problems, which makes it important to think about which functions to actually implement. Even if they are very helpful in detecting the lanes it might not be worth it to implement them as they might worsen the FPS by too much.

### Problems which could not be solved

We did not manage to come even close to a solution for the harder challenge. This was due to the stark differences in brightness, which completely disrupted our lane detection. It might have been possible to change some parameters so that the lane detection would perform better for the harder challenge video, but it is likely that the lane detection would have performed worse for the other videos in return.

In general Thresholds etc. were chosen specifically so that they work for the first two videos and the kitti images. Due to this, a lot of the parameters chosen for the lane detection are somewhat "overfitted" and the lane detection would probably not work as well for new videos which were not tested during development. For a lane detection that works for more universally, it would have been necessary to test it on a wide array of different videos instead of just ensuring that it works well for two specific ones.

### Outlook

Thresholding could be improved further and would probably lead to better lane detection. For example, one way to improve it would be to seperate the window in two parts so that the thresholds can be applied locally.

## :memo: License

This project is licensed under [MIT](https://github.com/felixhoffmnn/digitale-bildverarbeitung/blob/main/LICENSE).

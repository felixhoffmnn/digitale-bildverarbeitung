import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Fix operations (typing)
# TODO: Add docstring
def thresh_img(img: cv.Mat) -> cv.Mat:
    colors = {
        "yellow": {"min": np.array([150, 150, 0]), "max": np.array([255, 255, 120])},
        "white": {"min": np.array([185, 185, 185]), "max": np.array([255, 255, 255])},
    }
    saturation_threshold = (180, 200)
    sobel_x_threshold = (40, 100)
    hsv_s_threshold = (200, 215)

    # Isolate the blue channel for sobel filtering, so the yellow line is easily detected
    color_channel = img[:, :, 2]

    # Convert to HLS color space and isolate the saturation channel
    hls_channel = cv.cvtColor(img, cv.COLOR_RGB2HLS)[:, :, 2]

    # Convert to HVS color space and isolate the light channel
    hsv_channel = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, 1]

    yellow_mask = cv.inRange(img, colors["yellow"]["min"], colors["yellow"]["max"])
    white_mask = cv.inRange(img, colors["white"]["min"], colors["white"]["max"])

    # Apply Sobel x filtering on the blue color channel
    sobelx = cv.Sobel(color_channel, cv.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Apply the threshold on the x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_x_threshold[0]) & (scaled_sobel <= sobel_x_threshold[1])] = 1

    # Apply the threshold on the hls saturation channel
    hls_s_binary = np.zeros_like(hls_channel)
    hls_s_binary[(hls_channel >= saturation_threshold[0]) & (hls_channel <= saturation_threshold[1])] = 1

    # Apply the threshold on the hsv saturation channel
    hsv_s_binary = np.zeros_like(hsv_channel)
    hsv_s_binary[(hsv_channel >= hsv_s_threshold[0]) & (hsv_channel <= hsv_s_threshold[1])] = 1

    # Apply the threshold on the yellow mask
    yellow_binary = (yellow_mask // 255).astype(np.uint8)

    # Apply the threshold on the white mask
    white_binary = (white_mask // 255).astype(np.uint8)

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[
        (hls_s_binary == 1) | (sxbinary == 1) | (hsv_s_binary == 1) | (yellow_binary == 1) | (white_binary == 1)
    ] = 1

    return combined_binary

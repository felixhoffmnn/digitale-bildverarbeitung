import cv2 as cv
import numpy as np


# TODO: Refactor this function
# TODO: Fix operations (typing)
# TODO: Add docstring
def thresh_img(img: cv.Mat) -> cv.Mat:
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # TODO: Check hsv values
    colors = {
        "yellow": {"min": np.array([0, 15, 215]), "max": np.array([179, 50, 255])},
        "white": {"min": np.array([15, 50, 225]), "max": np.array([35, 225, 255])},
    }
    yellow_mask = cv.inRange(img_hsv, colors["yellow"]["min"], colors["yellow"]["max"])
    white_mask = cv.inRange(img_hsv, colors["white"]["min"], colors["white"]["max"])
    color_mask = cv.bitwise_or(yellow_mask, white_mask)
    color_mask[color_mask > 0] = 1

    # Apply Sobel x filtering on the blue color channel
    color_channel = img[:, :, 2]  # Isolate the blue channel for sobel filtering, so the yellow line is easily detected
    sobelx = cv.Sobel(color_channel, cv.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Apply the threshold on the x gradient
    sobel_x_threshold = np.array([40, 100])
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_x_threshold[0]) & (scaled_sobel <= sobel_x_threshold[1])] = 1
    sobel_mask = cv.bitwise_or(sxbinary, color_mask)

    # Apply the threshold on the hls saturation channel
    saturation_threshold = np.array([180, 200])
    hls_channel = cv.cvtColor(img, cv.COLOR_RGB2HLS)[
        :, :, 2
    ]  # Convert to HLS color space and isolate the saturation channel
    hls_s_binary = np.zeros_like(hls_channel)
    hls_s_binary[(hls_channel >= saturation_threshold[0]) & (hls_channel <= saturation_threshold[1])] = 1
    hls_mask = cv.bitwise_or(hls_s_binary, sobel_mask)

    # Apply the threshold on the hsv saturation channel
    hsv_s_threshold = np.array([200, 215])
    hsv_channel = cv.cvtColor(img, cv.COLOR_RGB2HSV)[
        :, :, 1
    ]  # Convert to HVS color space and isolate the light channel
    hsv_s_binary = np.zeros_like(hsv_channel)
    hsv_s_binary[(hsv_channel >= hsv_s_threshold[0]) & (hsv_channel <= hsv_s_threshold[1])] = 1
    hsv_mask = cv.bitwise_or(hsv_s_binary, hls_mask)

    filtered_image = cv.bitwise_and(img_gray, img_gray, mask=hsv_mask)

    return filtered_image

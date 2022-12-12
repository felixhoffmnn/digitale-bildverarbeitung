# Perspective

While the images and videos show the whole view of the camera in front of the car, for detecting the lane we are only interested in the part of the image where the road and, more specifically, our lane is. Therefore, we select a region of interest (ROI) in the shape of a trapezoid in the image and only process the pixels in this region. In this way, we can reduce possible noise by ignore the pixels in the sky, trees, cars, etc. and focus on the pixels that are most likely to be part of the lane.

In addition, to be able to determine the curvature of the lines, we have to transform the image to a bird's eye view. This is done by applying a perspective transformation to the image. The transformation is done by defining the source and destination points of the transformation. In the implementation we are using the OpenCV function `cv2.getPerspectiveTransform()` and the inverse transformation is done by using the OpenCV function `cv2.warpPerspective()`.

::: src.pipeline.perspective

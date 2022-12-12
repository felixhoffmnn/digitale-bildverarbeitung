# Lane Detection and Curve Fitting

The following functions are used to detect the lane lines. For one, lines can be detected using sliding windows. For the other, the lines are detected using the previous lines as a starting point.

To prevent the algorithm from detecting a line in the middle of the road, we use a offset from set midpoint when getting the starting points for the sliding windows.

Moreover, for every new live we calculate the Mean Squared Error (MSE) between the new line and the last one as a measure of discrepancy. If the MSE is too high (> 205), we use the last line instead of the new one as the new line does not seem to be valid.

In case there are many consecutive frames with a high MSE (n = 30), we try to detect the lines using the sliding windows technique again. This is only done, if the image is not too dark (sum of all pixels > 200000) because then the line detection based on sliding windows would probably not be successful.

!!! note "Source"

    As a starting point we used an existing implementation of the curve calculation. But for this project we adapted and extended the code to our needs including extensions for performance improvements and handling of difficulties like shadows, changing light conditions and missing lines.

    The original code can be found here

    - https://github.com/ndrplz/self-driving-car/blob/2bdcc7c822e8f03adc0a7490f1ae29a658720713/project_4_advanced_lane_finding/line_utils.py

::: src.pipeline.lane

# Threshold

Thresholding is a type of image segmentation. Using it, pixels are supposed to be changed so that the image becomes easier to analyze. In this pipeline, this is done by first converting the image to different colorspaces and then applying a sobel filter to the colorspaces, which leads to an output which easiert to process in the following steps.

::: src.pipeline.threshold

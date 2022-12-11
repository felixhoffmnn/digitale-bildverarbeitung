# def prepare_out_blend_frame(undist, thresh, transform):  # line_lt, line_rt, img_fit
#     """
#     Prepare the final pretty pretty output blend, given all intermediate pipeline images
#     :param blend_on_road: color image of lane blend onto the road
#     :param img_binary: thresholded binary image
#     :param img_birdeye: bird's eye view of the thresholded binary image
#     :param img_fit: bird's eye view with detected lane-lines highlighted
#     :param line_lt: detected left lane-line
#     :param line_rt: detected right lane-line
#     :param offset_meter: offset from the center of the lane
#     :return: pretty blend with all images and stuff stitched


#     Source: https://github.com/sidroopdaska/SelfDrivingCar/blob/master/AdvancedLaneLinesDetection/lane_tracker.ipynb
#     """
#     h, w = undist.shape[:2]

#     thumb_ratio = 0.2
#     thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

#     off_x, off_y = 20, 15

#     # add a gray rectangle to highlight the upper area
#     mask = undist.copy()
#     mask = cv.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h + 2 * off_y), color=(0, 0, 0), thickness=cv.FILLED)
#     undist = cv.addWeighted(src1=mask, alpha=0.2, src2=undist, beta=0.8, gamma=0)

#     # add thumbnail of binary image
#     thumb_binary = cv.resize(thresh, dsize=(thumb_w, thumb_h))
#     thumb_binary = cv.threshold(thumb_binary, 0, 255, cv.THRESH_BINARY)[1]
#     undist[off_y : thumb_h + off_y, off_x : off_x + thumb_w, :] = thumb_binary

#     # add thumbnail of bird's eye view
#     thumb_birdeye = cv.resize(transform, dsize=(thumb_w, thumb_h))
#     thumb_birdeye = cv.threshold(thumb_birdeye, 0, 255, cv.THRESH_BINARY)[1]
#     undist[off_y : thumb_h + off_y, 2 * off_x + thumb_w : 2 * (off_x + thumb_w), :] = thumb_birdeye

#     # add thumbnail of bird's eye view (lane-line highlighted)
#     # thumb_img_fit = cv.resize(img_fit, dsize=(thumb_w, thumb_h))
#     # undist[off_y : thumb_h + off_y, 3 * off_x + 2 * thumb_w : 3 * (off_x + thumb_w), :] = thumb_img_fit

#     # add text (curvature and offset info) on the upper right of the blend
#     # mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
#     # font = cv.FONT_HERSHEY_SIMPLEX
#     # cv.putText(
#     #     blend_on_road,
#     #     "Curvature radius: {:.02f}m".format(mean_curvature_meter),
#     #     (860, 60),
#     #     font,
#     #     0.9,
#     #     (255, 255, 255),
#     #     2,
#     #     cv.LINE_AA,
#     # )
#     # cv.putText(
#     #     blend_on_road,
#     #     "Offset from center: {:.02f}m".format(offset_meter),
#     #     (860, 130),
#     #     font,
#     #     0.9,
#     #     (255, 255, 255),
#     #     2,
#     #     cv.LINE_AA,
#     # )

#     return undist

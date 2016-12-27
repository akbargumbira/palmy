# coding=utf-8
import numpy as np
import cv2
from extract_image import extract_image

images = []
for i in range(9):
    img_src = 'test_palm%s.jpg' % (i+1)
    im = cv2.resize(cv2.imread(img_src), (300, 300))
    palm_contour, palm_center, palm_size, fingertips = extract_image(im)
    # cv2.drawContours(im, [palm_contour], -1, (0, 255, 0), 3)
    cv2.circle(im, palm_center, 5, (0, 0, 255), 5)
    cv2.circle(im, palm_center, palm_size/2, (255, 255, 255), 2)
    for i, fingertip in enumerate(fingertips):
        cv2.circle(im, fingertip, 5, (0, 0, 255), 5)
        cv2.line(im, palm_center, fingertip, (255, 255, 255), 2)
        cv2.putText(
            im, str(i), (fingertip[0]-10, fingertip[1]+10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
            (255, 255, 255), 2)
        # cv2.putText(
        #     im, str(int(angles[i])), (fingertip[0]-30, fingertip[1]+10),
        #                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
        #     (255, 255, 255), 2)
    images.append(im)

# # Draw palm rectangle
# left_upper = (int(palm_center1[0] - palm_size1/2), int(palm_center1[1]-palm_size1/2))
# right_bottom = (int(palm_center1[0] + palm_size1/2), int(palm_center1[1]+palm_size1/2))
# cv2.rectangle(im1, left_upper, right_bottom, (0,255,0), 3)

# Draw convex hull
# cv2.polylines(im1, np.int32([hull1]), 1, (0,0,255), 3)


cv2.imshow("images", np.vstack(
    [np.hstack([images[0],images[1],images[2]]),
     np.hstack([images[3],images[4],images[5]]),
     np.hstack([images[6],images[7],images[8]])]))
# cv2.imshow("Output", im1)
cv2.waitKey(0)

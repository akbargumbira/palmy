# coding=utf-8
import math
import numpy as np
import cv2
from extract_image import extract_image, angle

im1 = cv2.imread("my_palm.jpg")
im2 = cv2.imread("palm1.jpg")
im3 = cv2.imread("palm2.jpg")
palm_contour1, palm_center1, palm_size1, fingertips1 = extract_image(im1)
palm_contour2, palm_center2, palm_size2, fingertips2 = extract_image(im2)
palm_contour3, palm_center3, palm_size3, fingertips3 = extract_image(im3)

# Draw contour
cv2.drawContours(im1, [palm_contour1], -1, (0, 255, 0), 3)
cv2.drawContours(im2, [palm_contour2], -1, (0, 255, 0), 3)
cv2.drawContours(im3, [palm_contour3], -1, (0, 255, 0), 3)

# Draw palm center
cv2.circle(im1, palm_center1, 5, (0, 0, 255), 5)
cv2.circle(im2, palm_center2, 5, (0, 0, 255), 5)
cv2.circle(im3, palm_center3, 5, (0, 0, 255), 5)

# Draw palm rectangle
left_upper = (int(palm_center1[0] - palm_size1/2), int(palm_center1[1]-palm_size1/2))
right_bottom = (int(palm_center1[0] + palm_size1/2), int(palm_center1[1]+palm_size1/2))
cv2.rectangle(im1, left_upper, right_bottom, (0,255,0), 3)

# Draw fingertips
for fingertip in fingertips1:
    cv2.circle(im1, fingertip, 5, (0, 0, 255), 5)
for fingertip in fingertips2:
    cv2.circle(im2, fingertip, 5, (0, 0, 255), 5)
for fingertip in fingertips3:
    cv2.circle(im3, fingertip, 5, (0, 0, 255), 5)

# Draw convex hull
# cv2.polylines(im1, np.int32([hull1]), 1, (0,0,255), 3)


# # Contour centroid
# M = cv2.moments(palm_contour1)
# centroid_x = int(M['m10']/M['m00'])
# centroid_y = int(M['m01']/M['m00'])
# cv2.circle(im1,(centroid_x, centroid_y),5,[75,21,25],-1)

#
# (x,y),radius = cv2.minEnclosingCircle(palm_contour1[far_defects])
# center = (int(x),int(y))
# radius = int(radius)
# cv2.circle(im1,center,radius,(0,255,0),2)

cv2.imshow("images", np.hstack([im1, im2, im3]))
# cv2.imshow("Output", im1)
cv2.waitKey(0)

# coding=utf-8
import numpy as np
import cv2
from extract_image import extract_image

im1 = cv2.imread("my_palm.jpg")
im2 = cv2.imread("palm1.jpg")
im3 = cv2.imread("palm2.jpg")
palm_contour1 = extract_image(im1)
palm_contour2 = extract_image(im2)
palm_contour3 = extract_image(im3)

cv2.drawContours(im1, [palm_contour1], -1, (0, 255, 0), 3)
cv2.drawContours(im2, [palm_contour2], -1, (0, 255, 0), 3)
cv2.drawContours(im3, [palm_contour3], -1, (0, 255, 0), 3)

cv2.imshow("images", np.hstack([im1, im2, im3]))
cv2.waitKey(0)

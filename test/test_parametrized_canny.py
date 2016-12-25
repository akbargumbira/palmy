# coding=utf-8
import numpy as np
import cv2
from parametrized_canny import parametrized_canny

im1 = cv2.imread("my_palm.jpg")
im2 = cv2.imread("my_palm2.jpg")
im3 = cv2.imread("palm1.jpg")
im4 = cv2.imread("palm2.jpg")

im1 = parametrized_canny(im1)
im2 = parametrized_canny(im2)
im3 = parametrized_canny(im3)
im4 = parametrized_canny(im4)
cv2.imshow("images", np.hstack([im1, im2, im3, im4]))
cv2.waitKey(0)

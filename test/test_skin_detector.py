# coding=utf-8
import numpy as np
import cv2
from skin_detector import detect_skin

# Read Image
im1 = cv2.resize(cv2.imread("test_palm1.jpg"), (300, 300))
im2 = cv2.resize(cv2.imread("test_palm2.jpg"), (300, 300))
im3 = cv2.resize(cv2.imread("test_palm3.jpg"), (300, 300))
im4 = cv2.resize(cv2.imread("test_palm4.jpg"), (300, 300))
im5 = cv2.resize(cv2.imread("test_palm5.jpg"), (300, 300))
skin1 = detect_skin(im1)
skin2 = detect_skin(im2)
skin3 = detect_skin(im3)
skin4 = detect_skin(im4)
skin5 = detect_skin(im5)
cv2.imshow("images", np.hstack([skin1, skin2, skin3, skin4, skin5]))
cv2.waitKey(0)

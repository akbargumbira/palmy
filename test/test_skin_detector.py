# coding=utf-8
import numpy as np
import cv2
from skin_detector import detect_skin

# Read Image
im1 = cv2.imread("my_palm.jpg")
im2 = cv2.imread("palm1.jpg")
im3 = cv2.imread("palm2.jpg")
skin1 = detect_skin(im1)
skin2 = detect_skin(im2)
skin3 = detect_skin(im3)
cv2.imshow("images", np.hstack([skin1, skin2, skin3]))
cv2.waitKey(0)

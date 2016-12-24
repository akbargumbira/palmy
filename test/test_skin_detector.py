import numpy as np
import cv2
from skin_detector import detect_skin

lower = np.array([0, 48, 20], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

# Read Image
im = cv2.imread("my_palm.jpg")

cv2.imshow("Output", detect_skin(im))
cv2.waitKey(0)

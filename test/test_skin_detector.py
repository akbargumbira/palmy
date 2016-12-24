import numpy as np
import cv2
from skin_detector import detect_skin

# Read Image
im = cv2.imread("my_palm.jpg")

cv2.imshow("Output", detect_skin(im))
cv2.waitKey(0)

import numpy as np
import cv2
from skin_detector import detect_skin

# Read Image
im = cv2.imread("my_palm.jpg")

# Just use the skin
skin_im = detect_skin(im)

# Get canny edges from the skin image
# skin_edged = cv2.Canny(im, 50, 100)

# Find contours, get the largest one
(_, contours, _) = cv2.findContours(
    skin_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
palm_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

cv2.drawContours(im, [palm_contour], -1, (0, 255, 0), 3)
cv2.imshow("Output", im)
cv2.waitKey(0)

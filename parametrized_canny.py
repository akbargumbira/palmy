import os
import cv2
import numpy as np


def parametrized_canny(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    v = np.median(blurred)
    sigma = 0.33
    min_threshold = int(max(0, (1 - sigma) * v))
    max_threshold = int(min(255, (1.0 + sigma) * v))
    image = cv2.Canny(blurred, min_threshold, max_threshold)
    return image

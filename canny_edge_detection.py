import os
import cv2
import numpy as np


class CannyEdgeDetection(object):
    MINIMUM_THRESHOLD = 50
    MAXIMUM_THRESHOLD = 200

    def __init__(self, img_src):
        self._img_src = img_src
        self.img = cv2.imread(self._img_src, 0)
        self.edges = cv2.Canny(
            self.img,
            CannyEdgeDetection.MINIMUM_THRESHOLD,
            CannyEdgeDetection.MAXIMUM_THRESHOLD)

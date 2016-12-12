import os
import cv2
import numpy as np


class CannyEdgeDetection(object):
    def __init__(self, img_src):
        self._img_src = img_src
        self.img = cv2.imread(self._img_src, 0)
        self.edges = cv2.Canny(self.img, 50, 80)

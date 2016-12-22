import os
import cv2
from cv2 import Point
import numpy as np
import math


class PalmModel(object):
    def __init__(
            self,
            palm_center=None,
            fingertips_location=None):
        if palm_center:
            self.palm_center = palm_center
        if fingertips_location:
            self.fingertips_location = fingertips_location

        # Fingertips location
        if self.fingertips_location:
            self.thumb_location = self.fingertips_location[0]
            self.index_location = self.fingertips_location[1]
            self.middle_location = self.fingertips_location[2]
            self.ring_location = self.fingertips_location[3]
            self.baby_location = self.fingertips_location[4]

        self.fingertips_angles = []
        self.define_finger_angles()

    def define_finger_angles(self):
        """Redefine finger angles using palm center and fingertips location."""
        for i, finger_location in self.fingertips_location:
            dx = self.palm_center[0] - finger_location[0]
            dy = self.palm_center[1] - finger_location[1]
            r = math.sqrt(dx**2 + dy**2)
            angle = math.degrees(math.acos(dx/r))
            self.fingertips_angles[i] = angle


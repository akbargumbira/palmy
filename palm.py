# coding=utf-8
import math
import cv2
from utilities import rotate_points, rotate_point

class Palm(object):
    def __init__(
            self,
            fingertips_location=None,
            palm_center=(0, 0),
            palm_radius=0,
            palm_rotation=0):
        if fingertips_location:
            self.fingertips_location = fingertips_location
        self.palm_center = palm_center
        self.palm_radius = palm_radius
        self.palm_rotation = palm_rotation

        # Fingertips location
        if self.fingertips_location:
            self.thumb_location = self.fingertips_location[0]
            self.index_location = self.fingertips_location[1]
            self.middle_location = self.fingertips_location[2]
            self.ring_location = self.fingertips_location[3]
            self.baby_location = self.fingertips_location[4]

        self.palm_height = 1.6 * self.palm_radius
        self.palm_width = 1.5 * self.palm_radius

    def draw(self, image):
        """Draw the palm features on the image."""
        # Draw palm center
        cv2.circle(image, self.palm_center, 3, (0, 255, 0), -1)

        # Draw center bottom to fingertips
        palm_center_bottom = (
            self.palm_center[0],
            self.palm_center[1] + int(self.palm_height/2))
        palm_center_bottom = rotate_point(
            palm_center_bottom, self.palm_center, self.palm_rotation)
        for finger in self.fingertips_location:
            cv2.circle(image, finger, 3, (255, 255, 255), -1)
            cv2.line(image, palm_center_bottom, finger, (255, 255, 255), 2)

        # Draw palm rectangle
        palm_left_upper = (
            self.palm_center[0] - (self.palm_width/2),
            self.palm_center[1] - (self.palm_height/2))
        palm_left_bottom = (
            self.palm_center[0] - (self.palm_width/2),
            self.palm_center[1] + (self.palm_height/2))
        palm_right_upper = (
            self.palm_center[0] + (self.palm_width/2),
            self.palm_center[1] - (self.palm_height/2))
        palm_right_bottom = (
            self.palm_center[0] + (self.palm_width/2),
            self.palm_center[1] + (self.palm_height/2))
        points = [palm_left_upper, palm_right_upper, palm_right_bottom, palm_left_bottom]
        points = rotate_points(points, self.palm_center, self.palm_rotation)
        for i, point in enumerate(points):
            cv2.line(
                image, points[i], points[int((i+1) % 4)], (255, 255, 255), 2)

        return image


# Defining Palm Model here
model_fingertips = [(12, 325), (174, 48), (278, 12), (368, 38),  (458, 126)]
model_palm_center = (310, 340)
model_palm_radius = 140
model_palm_rotation = 0
PalmModel = Palm(
    model_fingertips, model_palm_center, model_palm_radius, model_palm_rotation)


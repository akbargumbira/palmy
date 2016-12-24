# coding=utf-8
import numpy as np
import cv2


def detect_skin(input_image):
    lower = np.array([0, 40, 20], dtype="uint8")
    upper = np.array([180, 255, 255], dtype="uint8")

    converted = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(input_image, input_image, mask=skin_mask)
    skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    return skin


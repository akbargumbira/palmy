# coding=utf-8
import numpy as np
import cv2


def detect_skin(input_image):
    lower = np.array([0, 40, 120], dtype="uint8")
    upper = np.array([180, 255, 255], dtype="uint8")

    converted = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(input_image, input_image, mask=skin_mask)
    skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    # skin_ycrcb_mint = np.array((0, 133, 77))
    # skin_ycrcb_maxt = np.array((255, 173, 127))
    #
    # im_ycrcb = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCR_CB)
    # skin = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    return skin


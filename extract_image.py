# coding=utf-8
import cv2
from skin_detector import detect_skin


def extract_image(input_image):
    # Just use the skin
    skin_image = detect_skin(input_image)

    # Get canny edges from the skin image
    # i_im = cv2.Canny(im, 50, 200)

    # Find contours, get the largest one
    #  TODO: I should consider taking more than 1 contours (because there
    #       could be more than 1 palm in the image)
    (_, contours, _) = cv2.findContours(
        skin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    palm_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Approximate for better contour
    epsilon = 0.001*cv2.arcLength(palm_contour, True)
    palm_contour = cv2.approxPolyDP(palm_contour, epsilon, True)

    return palm_contour

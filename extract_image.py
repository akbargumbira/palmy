# coding=utf-8
import math
import cv2
import numpy as np
from skin_detector import detect_skin
from parametrized_canny import parametrized_canny
import sys


def extract_image(input_image):
    # Just use the skin
    image = detect_skin(input_image)

    # Use canny (alternative)
    # image = parametrized_canny(input_image)

    # Find contours, get the largest one
    #  TODO: I should consider taking more than 1 contours (because there
    #       could be more than 1 palm in the image)
    (_, contours, _) = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    palm_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Approximate for better contour
    epsilon = 0.0015*cv2.arcLength(palm_contour, True)
    palm_contour = cv2.approxPolyDP(palm_contour, epsilon, True)

    # Get fingertips
    fingertips, gaps = get_fingertips(palm_contour, input_image)

    # Get the palm center
    palm_center = find_palm_center(input_image, palm_contour, gaps)

    # Palm radius (the center is better from above)
    (_, _), radius = cv2.minEnclosingCircle(np.asarray(gaps))
    radius = int(radius)
    cv2.circle(input_image, palm_center, radius, (0, 255, 0), 2)
    palm_size = 2 * radius

    return palm_contour, palm_center, palm_size, fingertips


def find_palm_center(input_image, palm_contour, gaps):
    M = cv2.moments(palm_contour)
    moment = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    center = moment
    cv2.circle(input_image, center, 5, (0, 255, 0), 5)
    # Create bounding with center the max_point and find better center
    kernel_size = 5
    center_moved = True
    max_distance = 0
    while center_moved:
        for x in range((-kernel_size+1)/2, (kernel_size+1)/2):
            for y in range((-kernel_size+1)/2, (kernel_size+1)/2):
                dist = cv2.pointPolygonTest(
                    palm_contour, (center[0]+x, center[1]+y), True)
                if dist > max_distance:
                    max_distance = dist
                    max_point = (center[0]+x, center[1]+y)
        center_moved = center != max_point
        center = max_point
        cv2.circle(input_image, center, 2, (255, 255, 255), 2)

    return center

def angle(s, f, e):
    sf = math.hypot(f[0]-s[0], f[1]-s[1])
    ef = math.hypot(f[0]-e[0], f[1]-e[1])
    dot = (s[0]-f[0])*(e[0]-f[0]) + (s[1]-f[1])*(e[1]-f[1])
    angle = math.degrees(math.acos(dot/(sf*ef)))
    return angle


def get_fingertips(palm_contour, im):
    rect = cv2.minAreaRect(palm_contour)
    palm_width = rect[1][0]
    palm_height = rect[1][1]

    hull = cv2.convexHull(palm_contour, returnPoints=False)
    defects = cv2.convexityDefects(palm_contour, hull)

    max_distance = 0.8 * palm_height
    min_distance = 0.1 * palm_height
    min_angle = 5
    max_angle = 100

    fingertips = []
    gaps = []
    min_x_fingertips = sys.maxint
    last_fingertips = (0, 0)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(palm_contour[s][0])
        end = tuple(palm_contour[e][0])
        far = tuple(palm_contour[f][0])

        dist_sf = math.hypot(far[0] - start[0], far[1] - start[1])
        good_sf = min_distance < dist_sf and dist_sf < max_distance
        dist_ef = math.hypot(far[0] - end[0], far[1] - end[1])
        good_ef = min_distance < dist_ef and dist_ef < max_distance
        angle_esf = angle(start, far, end)
        good_angle = min_angle < angle_esf and angle_esf < max_angle

        good_finger_defects = good_sf and good_ef and good_angle
        if good_sf and good_ef:
            gaps.append(far)

        if good_finger_defects:
            fingertips.append(start)
            # cv2.line(im, start, far, (255, 0, 0), 3)
            # cv2.line(im, far, end, (255, 0, 0), 3)
            cv2.circle(im, far, 5, (255, 255, 255), 5)

            if start[0] < min_x_fingertips:
                min_x_fingertips = start[0]
                last_fingertips = end

    fingertips.append(last_fingertips)
    return fingertips, gaps

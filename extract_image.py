# coding=utf-8
import math
import itertools
import cv2
import numpy as np
from skin_detector import detect_skin


def extract_image(input_image):
    # Just use the skin
    image = detect_skin(input_image)

    # Use canny (alternative)
    # image = parametrized_canny(image)

    # Find contours, get the largest one
    #  TODO: I should consider taking more than 1 contours (because there
    #       could be more than 1 palm in the image)
    (_, contours, _) = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    palm_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Approximate for better contour
    epsilon = 0.0001*cv2.arcLength(palm_contour, True)
    palm_contour = cv2.approxPolyDP(palm_contour, epsilon, True)

    # Get fingertips
    fingertips, palm_center, gaps = get_fingertips(palm_contour, input_image)

    # Palm radius (the center is better from above)
    (_, _), radius = cv2.minEnclosingCircle(np.asarray(gaps))
    radius = int(radius)
    # cv2.circle(input_image, palm_center, radius, (0, 255, 0), 2)
    palm_size = 2 * radius

    return palm_contour, palm_center, palm_size, fingertips


def find_palm_center(input_image, palm_contour, gaps):
    M = cv2.moments(palm_contour)
    moment = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    center = moment
    # cv2.circle(input_image, center, 5, (0, 255, 0), 5)
    # Create bounding with center the max_point and find better center
    kernel_size = 5
    center_moved = True
    max_distance = 0
    max_point = (0,0)
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
        # cv2.circle(input_image, center, 2, (255, 255, 255), 2)

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
    palm_box_size = min(palm_width, palm_height)

    hull = cv2.convexHull(palm_contour, returnPoints=False)
    defects = cv2.convexityDefects(palm_contour, hull)

    max_distance = 0.8 * palm_box_size
    min_distance = 0.1 * palm_box_size
    min_angle = 0
    max_angle = 90

    gaps = []
    filtered_start_end = []
    j = 0
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

        good_defects = good_sf and good_ef and good_angle
        if good_sf and good_ef:
            gaps.append(far)

        if good_defects:
            j += 1
            filtered_start_end.append(start)
            filtered_start_end.append(end)
            # cv2.line(im, start, far, (255, 0, 0), 3)
            # cv2.line(im, far, end, (0, 0, 255), 3)
            # cv2.circle(im, far, 5, (255, 255, 255), 5)

    # Get the palm center
    palm_center = find_palm_center(im, palm_contour, gaps)

    min_fingertips_distance = 0.07 * palm_box_size
    closed_points = []
    combs = list(itertools.combinations(range(len(filtered_start_end)), 2))
    for comb in combs:
        dist = math.hypot(
            filtered_start_end[comb[0]][0] - filtered_start_end[comb[1]][0],
            filtered_start_end[comb[0]][1] - filtered_start_end[comb[1]][1])
        if dist < min_fingertips_distance:
            closed_points.append(filtered_start_end[comb[0]])

    fingertips = filtered_start_end
    for point in closed_points:
        fingertips.remove(point)

    # Sort fingertips (thumb, index, ..., baby)
    angles360 = []
    angles180 = []
    for i, finger_location in enumerate(fingertips):
        dx = palm_center[0] - finger_location[0]
        dy = palm_center[1] - finger_location[1]
        # r = math.sqrt(dx ** 2 + dy ** 2)
        angle360 = math.degrees(math.atan2(dy, dx))
        angle180 = math.degrees(math.atan2(dy, dx))
        # ref_point = (0, palm_center[1])
        # angle = math.atan2()
        # angle = angle * 360 / (2*3.14)
        if angle360 < 0:
            angle360 += 360
        angles360.append(angle360)
        angles180.append(angle180)
    if max(angles360) - min(angles360) < 180:
        idx = sorted(range(len(angles360)), key=lambda k: angles360[k])
        # return
    else:
        idx = sorted(range(len(angles180)), key=lambda k: angles180[k])

    fingertips = [fingertips[i] for i in idx]
    return fingertips, palm_center, gaps



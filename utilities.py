# coding=utf-8
import math


def rotate_point(point, center, angle):
    x = int(
            ((point[0] - center[0]) * math.cos(math.radians(angle))) -
            ((point[1] - center[1]) * math.sin(math.radians(angle)))
        ) + center[0]
    y = int(
            ((point[0] - center[0]) * math.sin(math.radians(angle))) +
            ((point[1] - center[1]) * math.cos(math.radians(angle)))
        ) + center[1]

    return x, y


def rotate_points(points, center, angle):
    result = []
    for point in points:
        result.append(rotate_point(point, center, angle))

    return result

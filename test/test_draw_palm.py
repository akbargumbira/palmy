# coding=utf-8
import numpy as np
import cv2
from extract_image import extract_image
from palm import Palm

images = []
for i in range(9):
    img_src = 'images/test_palm%s.jpg' % (i+1)
    im = cv2.resize(cv2.imread(img_src), (300, 300))
    palm_contour, palm_center, palm_radius, fingertips, rotation = extract_image(im)
    palm = Palm(fingertips, palm_center, palm_radius, rotation)
    images.append(palm.draw(im))


cv2.imshow("images", np.vstack(
    [np.hstack([images[0],images[1],images[2]]),
     np.hstack([images[3],images[4],images[5]]),
     np.hstack([images[6],images[7],images[8]])]))

cv2.waitKey(0)

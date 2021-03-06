# coding=utf-8
import numpy as np
import cv2
from parametrized_canny import parametrized_canny

# Read Image
images = []
for i in range(9):
    img_src = 'images/test_palm%s.jpg' % (i + 1)
    image = cv2.resize(cv2.imread(img_src), (300, 300))
    canny = parametrized_canny(image)
    images.append(canny)

cv2.imshow("images", np.vstack(
    [np.hstack([images[0],images[1],images[2]]),
     np.hstack([images[3],images[4],images[5]]),
     np.hstack([images[6],images[7],images[8]])]))

cv2.waitKey(0)

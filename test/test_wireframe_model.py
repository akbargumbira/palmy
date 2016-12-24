import cv2
import numpy as np

# Read Image
im = cv2.imread("my_palm.jpg")
size = im.shape

# Fingers
finger_locations = np.array([
    (10, 325),  # Thumb
    (191, 43),  # Index
    (303, 11),  # Middle
    (379, 44),  # Ring
    (471, 137),  # Baby
], dtype="double")

# Palm
palm_center = (300, 340)
palm_size = 230
# Draw palm center
cv2.circle(im, (int(palm_center[0]), int(palm_center[1])), 3, (0, 0, 255), -1)
# Draw palm boundaries
palm_left_upper = (palm_center[0]-(palm_size/2), palm_center[1]-(palm_size/2))
palm_left_bottom = (palm_center[0]-(palm_size/2), palm_center[1]+(palm_size/2))
palm_right_upper = (palm_center[0]+(palm_size/2), palm_center[1]-(palm_size/2))
palm_right_bottom = (palm_center[0]+(palm_size/2), palm_center[1]+(palm_size/2))
cv2.line(im, palm_left_upper, palm_right_upper, (255, 0, 0), 2)
cv2.line(im, palm_right_upper, palm_right_bottom, (255, 0, 0), 2)
cv2.line(im, palm_right_bottom, palm_left_bottom, (255, 0, 0), 2)
cv2.line(im, palm_left_bottom, palm_left_upper, (255, 0, 0), 2)

# Circle fingertips and
palm_middle_bottom = (palm_center[0], palm_center[1]+(palm_size/2))
for p in finger_locations:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    cv2.line(im, palm_middle_bottom, (int(p[0]), int(p[1])), (255, 0, 0), 2)

# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)

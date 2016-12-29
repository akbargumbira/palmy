import cv2
from map_estimation import iterative_map

# Read Image
image = cv2.imread("images/test_palm1.jpg")
final_hypothesis = iterative_map(image)

# Display image
cv2.imshow("Output", final_hypothesis.draw(image))
cv2.waitKey(0)

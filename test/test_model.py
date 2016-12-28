import cv2
import numpy as np
from palm import PalmModel

# Read Image
image = cv2.imread("images/palm_model.jpg")
image = PalmModel.draw(image)

# Display image
cv2.imshow("Output", image)
cv2.waitKey(0)

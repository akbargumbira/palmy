# coding=utf-8
from matplotlib import pyplot
import cv2
from canny_edge_detection import CannyEdgeDetection

img_src = 'my_palm.jpg'
canny_edge_detection = CannyEdgeDetection(img_src)
cv2.imshow("Output", canny_edge_detection.edges)
cv2.waitKey(0)

# pyplot.subplot(121), pyplot.imshow(canny_edge_detection.img, cmap='gray')
# pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
# pyplot.subplot(122), pyplot.imshow(canny_edge_detection.edges, cmap='gray')
# pyplot.title('Edge image'), pyplot.xticks([]), pyplot.yticks([])
# pyplot.show()
#
# img_src = 'palm2.jpg'
# canny_edge_detection = CannyEdgeDetection(img_src)
#
# pyplot.subplot(121), pyplot.imshow(canny_edge_detection.img, cmap='gray')
# pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
# pyplot.subplot(122), pyplot.imshow(canny_edge_detection.edges, cmap='gray')
# pyplot.title('Edge image'), pyplot.xticks([]), pyplot.yticks([])
# pyplot.show()

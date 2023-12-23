import numpy as np
import argparse
import cv2
import imutils

image = imutils.loadImage()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

(T, threshInv) = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)


(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

cv2.imshow("Cats", cv2.bitwise_and(image, image, mask = thresh))
cv2.waitKey(0)
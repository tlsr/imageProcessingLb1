import numpy as np
import cv2
import imutils

image = imutils.loadImage()

cv2.imshow("Original", image)

cropped = image[184:275 , 98:204]
cv2.imshow("cat Face", cropped)
cv2.waitKey(0)
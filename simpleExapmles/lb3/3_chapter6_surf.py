import cv2
import imutils

img = imutils.loadImage()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
surf = cv2.SURF_create(8000)
keypoints, descriptor = surf.detectAndCompute(gray, None)
cv2.drawKeypoints(img, keypoints, img, (51, 163, 236),
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('surf_keypoints', img)
cv2.waitKey()
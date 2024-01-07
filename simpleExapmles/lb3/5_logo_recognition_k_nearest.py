import cv2
from matplotlib import pyplot as plt

img0 = cv2.imread('nasa-logo.jpg',cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('59_vab.jpg',cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2)
pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)
matches = [x[0] for x in pairs_of_matches
    if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]
# Draw the best 25 matches.
img_matches = cv2.drawMatches(
img0, kp0, img1, kp1, matches[:25], img1,
flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# Show the matches.
plt.imshow(img_matches)
plt.show()
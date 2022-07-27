# Morphological Transformations
# Erosion and Dilation
import cv2
import numpy as np
#
# img = cv2.imread('./images/j.png', 0)
# kernel = np.ones((5, 5), np.uint8)
# erosion = cv2.erode(img, kernel, iterations = 1)
# dilation = cv2.dilate(img, kernel, iterations = 1)
# cv2.imshow('erode', erosion)
# cv2.imshow('dilate', dilation)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Canny Edge Detection
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#
#     edges = cv2.Canny(frame, 10, 5)
#
#     cv2.imshow('Out', frame)
#     cv2.imshow('Canny Edges', edges)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# Corner Detection (Shape detection)
# import cv2
# import numpy as np
#
# img = cv2.imread('./images/shape2.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
#
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
# corners = np.int0(corners)
# for corner in corners:
#     x,y = corner.ravel()
#     cv2.circle(img, (x,y), 3, 245, -1)
# cv2.imshow('Corner_image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv2.imread('images/shape2.png')
# # print(img.shape)
# imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
# contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#
#
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
#     cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1] - 5
#     cv2.imshow('shapes', img)
#     cv2.waitKey(0)
#     if len(approx) == 3:
#         cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
#     elif len(approx) == 4 :
#         x, y , w, h = cv2.boundingRect(approx)
#         aspectRatio = float(w)/h
#         print(aspectRatio)
#         if aspectRatio >= 0.95 and aspectRatio < 1.05:
#             cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#         else:
#             cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 5:
#         cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 10:
#         cv2.putText(img, "star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     else:
#         cv2.putText(img, "circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#     cv2.imshow('shapes', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# laplacian Gradients
# First run
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#
#     laplacian = cv2.Laplacian(frame, cv2.CV_64F)
#     laplacian=cv2.resize(laplacian,(1280,960))
#
#     cv2.imshow('Out', frame)
#     cv2.imshow('Laplacian', laplacian)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# img_rgb = cv2.imread('./images/alpha1.jpg')
# laplacian = cv2.Laplacian(img_rgb, cv2.CV_64F)
# laplacian=cv2.resize(laplacian,(1280,960))
# cv2.imshow('Out',img_rgb)
# cv2.imshow('Laplacian', laplacian)
#
# cv2.waitKey(-1)
# cv2.destroyAllWindows()


#Second  run (getting slope of x and y)
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#
#     laplacian = cv2.Laplacian(frame, cv2.CV_64F)
#     slope_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
#     slope_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
#
#     cv2.imshow('Out', frame)
#     cv2.imshow('Laplacian', laplacian)
#     cv2.imshow('slopeX', slope_x)
#     cv2.imshow('slopey', slope_y)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
#Template Matching
# import cv2
# import numpy as np
#
# img_rgb = cv2.imread('./images/alpha1.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#
# template = cv2.imread('./images/k1.png', 0)
# w, h = template.shape[::-1]
#
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.9
#
# loc = np.where(res >= threshold)
#
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt,  (pt[0]+w, pt[1]+h), (0, 255, 255), 2)
#
# cv2.imshow('Detected', img_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Feature Matching
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# img1 = cv2.imread('./images/swami-vivekananda-1.jpg', 0)
# img2 = cv2.imread('./images/swami-vivekananda-1.jpg', 0)
# #orb (oriented Fast and Rotated Brief
# orb = cv2.ORB_create()
#
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
#
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# plt.imshow(img3)
# plt.show()

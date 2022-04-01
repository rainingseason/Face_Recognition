import numpy as np
import cv2
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
import KailunavatarCode.cannyfunction as cf


img = cv2.imread('lady1.JPG')
# print(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dim1 = (3000, 3000)
# resized = cv2.resize(gray_img, dim1)

# edges = cv2.Canny(resized,100,200)
result = convolve(gray_img, cf.gaussian_kernel())
(result, theta) = cf.sobel_filters(result)
# result = cf.non_max_suppression(result, theta)

# plt.subplot(121)
# plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(gray_img)
# plt.title('Result Image'), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.imshow("Original", img)
cv2.imshow("Gray", gray_img)
cv2.waitKey(0)
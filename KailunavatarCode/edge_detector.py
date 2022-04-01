import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import KailunavatarCode.cannyfunction as cf
import numpy as np
import os

# path = os.getcwd()
# os.chdir(path + "\\KailunavatarCode")

img = cv2.imread('lady1.JPG')
gray_img = cv2.imread('lady1.JPG', 0) # flag = 0 reads in grayscale
# img = cv2.imread('lady1.JPG')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pre-defined function
# edges = cv2.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

dim1 = (3000, 3000)
resized = cv2.resize(gray_img, dim1)

result = convolve(gray_img, cf.gaussian_kernel())
(result, theta) = cf.sobel_filters(result)
result = cf.non_max_suppression(result, theta)
# round_result = np.rint(result)

# plt.subplot(121)
# plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(result)
# plt.title('Result Image'), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.imshow("Original", img)
cv2.imshow("Result", round_result)
cv2.waitKey(0)
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

# threshold = 0.98
#
# thresholded = result / result.max()
# low_values_flags = thresholded < threshold  # values are below threshold
# thresholded[low_values_flags] = 0  # All low values set to 0
# high_values_flags = thresholded > threshold
# thresholded[high_values_flags] = 1

# round_result = np.rint(result)

# author's code
# img_smoothed = cf.convolve(resized, cf.gaussian_kernel())
# gradientMat, thetaMat = cf.sobel_filters(img_smoothed)
# nonMaxImg = cf.non_max_suppression(gradientMat, thetaMat)

# plt.subplot(121)
# plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(result)
# plt.title('Result Image'), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.imshow("Original", img)
cv2.imshow("Result", np.rint(result))
# cv2.imshow("Thresholded", thresholded)
# cv2.imshow("Author", nonMaxImg)
cv2.waitKey(0)

def sobel(img):
    Kx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    Ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)

    conv_img_x = 0
    conv_img_y = 0

    try:
        for row in range(len(img)):
            for column in range(len(img[row])):
                output_x = 0
                output_y = 0
                for pixel_row in range(3):
                    for pixel_column in range(3):
                        output_x += Kx[pixel_row][pixel_column] * img[row - pixel_row - 1][column - pixel_column - 1]
                        output_y += Ky[pixel_row][pixel_column] * img[row - pixel_row - 1][column - pixel_column - 1]
                conv_img_x[row][column] = output_x
                conv_img_y[row][column] = output_y

    except IndexError:
        pass
    return conv_img_x, conv_img_y
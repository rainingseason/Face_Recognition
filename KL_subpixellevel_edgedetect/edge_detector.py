import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import cannyfunction as cf

def toFitMatplotlib(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# pre-processing
path = r'C:/Users/kailu/PycharmProjects/face_reg/Assignment2/image/chessboard1.jpg'

img = cv2.imread(path) # BGR
print(img.shape)

# img = toFitMatplotlib(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dim1 = (400, 400)
gray_img = cv2.resize(gray_img, dim1)
print(gray_img.shape)
# print(gray_img.shape)

# dim1 = (3000, 3000)
# gray_img = cv2.resize(gray_img, dim1)

# gray_img, weak, strong = cf.threshold(gray_img)
# gray_img = cf.hysteresis(gray_img, weak)

# kernel = np.ones((3,3),np.uint8)
# gray_img = cv2.dilate(gray_img,kernel,iterations = 2) # dilate erode

gaussian_smooth = signal.convolve2d(gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')
# print(gaussian_smooth.shape)

(result, theta) = cf.sobel_filters(gaussian_smooth)
# print(result)

# subpixel level
result = cf.subpixel_function(result, theta)

# pixel level
# result = cf.non_max_suppression(result, theta)

result, weak, strong = cf.threshold(result)

result = cf.hysteresis(result, weak)

# subpixel part
# result = cv2.imwrite('subpixel.jpg', result)
# result = cv2.imread('subpixel.jpg')
# result = cv2.resize(result, (1600, 1600))
# before = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# result = cf.subpixel_function(before)
# print(result)


# , cmap='gray'
plt.subplot(121)
plt.imshow(gaussian_smooth, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(result, cmap='gray')
plt.title('Result Image')
plt.xticks([]), plt.yticks([])
plt.show()

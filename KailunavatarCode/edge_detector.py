import cv2
from scipy import signal
from matplotlib import pyplot as plt
import KailunavatarCode.cannyfunction as cf
import os

def toFitMatplotlib(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# pre-processing
path = os.getcwd()
path = path + '\\KailunavatarCode\\lady1.JPG'
img = cv2.imread(path) # BGR
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussian_smooth = signal.convolve2d(gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')

(result, theta) = cf.sobel_filters(gray_img)

result = cf.non_max_suppression(result, theta)
result, weak, strong = cf.threshold(result)

plt.subplot(121)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(result, cmap='gray')
plt.title('Result Image')
plt.xticks([]), plt.yticks([])
plt.show()
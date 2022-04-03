import cv2
from scipy import signal
from matplotlib import pyplot as plt
import cannyfunction as cf

def toFitMatplotlib(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# pre-processing
path = r'C:/Users/kailu/PycharmProjects/face_reg/Assignment2/image/lady1.JPG'

img = cv2.imread(path) # BGR
print(img.shape)
# dim1 = (300, 300)
# img = cv2.resize(img, dim1)
# img = toFitMatplotlib(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_img.shape)

# dim1 = (300, 300)
# img = cv2.resize(img, dim1)
# gray_img = cv2.resize(gray_img, dim1)

# edges = cv2.Canny(resized,100,200)
gaussian_smooth = signal.convolve2d(gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')
# print(gaussian_smooth.shape)

(result, theta) = cf.sobel_filters(gray_img)
# imgs_final = []
# imgs_final.append(result)
#
# for i, img in enumerate(imgs_final):
#     plt.imshow(img, cmap='gray')
# plt.show()

result = cf.non_max_suppression(result, theta)
result, weak, strong = cf.threshold(result)
# result = cf.hysteresis(result, weak)

# cv2.imshow('original', img)
# cv2.imshow('processed', gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# , cmap='gray'
plt.subplot(121)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(result, cmap='gray')
plt.title('Result Image')
plt.xticks([]), plt.yticks([])
plt.show()

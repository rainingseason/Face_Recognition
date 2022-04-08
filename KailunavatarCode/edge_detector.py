import cv2
from scipy import signal
from matplotlib import pyplot as plt
import KailunavatarCode.cannyfunction as cf
import os

def toFitMatplotlib(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# pre-processing
# path = os.getcwd()
path = 'C:\\Users\\Alyna Khoo Yi Jie\\Documents\\NTU\\Year 4\\Semester 2\\EE4208 INTELLIGENT SYSTEMS DESIGN\\Assignments\\Face Recognition'
# path = path + '\\KailunavatarCode\\lady1.JPG'
path = path + '\\Edge Detection\\faces_imgs\\Chess_board.jpeg'
# path = path + '\\Edge Detection\\faces_imgs\\Chessboard.jpeg'
# path = path + '\\Edge Detection\\faces_imgs\\Chessboard_Reference.png'
# path = path + '\\Edge Detection\\faces_imgs\\sunset.jpg'

# path = 'C:\\Users\\Alyna Khoo Yi Jie\\Documents\\NTU\\Year 4\\Semester 2\\EE4208 INTELLIGENT SYSTEMS DESIGN\\Assignments\\Face Recognition\\database\\alyna\\alyna_3.jpeg'

img = cv2.imread(path) # BGR
(dim_x, dim_y) = (img.shape[0], img.shape[1])
(small_dim_x, small_dim_y) = (dim_x//4, dim_y//4)
small_img = cv2.resize(img, (small_dim_x, small_dim_y))
gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

gaussian_smooth = signal.convolve2d(gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')

(result, theta) = cf.sobel_filters(gaussian_smooth)

result = cf.non_max_suppression(result, theta)
result, weak, strong = cf.threshold(result)
result = cf.hysteresis(result, weak)

cv2.imwrite('small_result.jpg', result)
result = cv2.imread('small_result.jpg')
result = cv2.resize(result, (dim_x, dim_y))

plt.subplot(121)
# plt.imshow(gray_img, cmap='gray')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(result, cmap='gray')
plt.title('Result Image')
plt.xticks([]), plt.yticks([])
plt.show()
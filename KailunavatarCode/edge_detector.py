import cv2
from scipy import signal
from matplotlib import pyplot as plt
import KailunavatarCode.cannyfunction as cf

path = 'C:\\Users\\Alyna Khoo Yi Jie\\Documents\\NTU\\Year 4\\Semester 2\\EE4208 INTELLIGENT SYSTEMS DESIGN\\Assignments\\Face Recognition'
# path = path + '\\KailunavatarCode\\lady1.JPG'
# path = path + '\\Edge Detection\\faces_imgs\\Chess_board.jpeg'
path = path + '\\Edge Detection\\faces_imgs\\Chessboard.jpeg'
# path = path + '\\Edge Detection\\faces_imgs\\Chessboard_Reference.png'
# path = path + '\\Edge Detection\\faces_imgs\\sunset.jpg'

def normal(path):
    # pre-processing
    img = cv2.imread(path)  # BGR
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gaussian_smooth = signal.convolve2d(gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')

    (result, theta) = cf.sobel_filters(gaussian_smooth)

    result = cf.non_max_suppression(result, theta)
    result, weak, strong = cf.threshold(result)
    result = cf.hysteresis(result, weak)
    return result, img, gray_img

def subpixel(gray_img):
    # pre-processing and resizing
    (dim_y, dim_x) = gray_img.shape
    (small_dim_x, small_dim_y) = (dim_x // 4, dim_y // 4)
    small_gray_img = cv2.resize(gray_img, (small_dim_x, small_dim_y))

    gaussian_smooth = signal.convolve2d(small_gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')
    (result, theta) = cf.sobel_filters(gaussian_smooth)

    result = cf.subpixel_func(result, theta)
    result, weak, strong = cf.threshold(result)
    result = cf.hysteresis(result, weak)
    return result, (small_dim_x, small_dim_y)

norm, img, gray_img = normal(path)
sub, (small_dim_x, small_dim_y) = subpixel(gray_img)
small_img = cv2.resize(img, (small_dim_x, small_dim_y))

plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(222)
plt.imshow(norm, cmap='gray')
plt.title('Original Edge Detector Result')
plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Reduced Image')
plt.xticks([]), plt.yticks([])

plt.subplot(224)
plt.imshow(sub, cmap='gray')
plt.title('Subpixel Edge Detector Result')
plt.xticks([]), plt.yticks([])
plt.show()
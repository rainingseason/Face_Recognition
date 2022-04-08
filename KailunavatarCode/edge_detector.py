import cv2
from scipy import signal
from matplotlib import pyplot as plt
import KailunavatarCode.cannyfunction as cf
import os

def toFitMatplotlib(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# path = os.getcwd()
path = 'C:\\Users\\Alyna Khoo Yi Jie\\Documents\\NTU\\Year 4\\Semester 2\\EE4208 INTELLIGENT SYSTEMS DESIGN\\Assignments\\Face Recognition'
# path = path + '\\KailunavatarCode\\lady1.JPG'
# path = path + '\\Edge Detection\\faces_imgs\\Chess_board.jpeg'
path = path + '\\Edge Detection\\faces_imgs\\Chessboard.jpeg'
# path = path + '\\Edge Detection\\faces_imgs\\Chessboard_Reference.png'
# path = path + '\\Edge Detection\\faces_imgs\\sunset.jpg'

def ced(gray_img):
    gaussian_smooth = signal.convolve2d(gray_img, cf.gaussian_kernel(), boundary='fill', mode='same')

    (result, theta) = cf.sobel_filters(gaussian_smooth)

    result = cf.non_max_suppression(result, theta)
    result, weak, strong = cf.threshold(result)
    result = cf.hysteresis(result, weak)
    return result

def normal(path):
    # pre-processing
    img = cv2.imread(path)  # BGR
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = ced(gray_img)
    return result, img, gray_img

def subpixel(gray_img):
    # pre-processing and resizing
    (dim_x, dim_y) = gray_img.shape
    (small_dim_x, small_dim_y) = (dim_x // 4, dim_y // 4)
    small_gray_img = cv2.resize(gray_img, (small_dim_x, small_dim_y))

    result = ced(small_gray_img)

    filename = 'small_result.jpg'
    cv2.imwrite(filename, result)
    result = cv2.imread(filename)
    if os.path.exists(filename):
        os.remove(filename)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.resize(result, (dim_y, dim_x)) # not sure why need to reverse the dimensions
    result = cf.find_true_edge(result)
    return result

norm, img, gray_img = normal(path)
sub = subpixel(gray_img)

plt.subplot(131)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(norm, cmap='gray')
plt.title('Normal Edge Detection')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(sub, cmap='gray')
plt.title('Subpixel Edge Detection')
plt.xticks([]), plt.yticks([])
plt.show()
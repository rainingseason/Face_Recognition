import numpy as np
from scipy import signal

# gaussian smoothing
def gaussian_kernel(sigma = 1.3):
    w = np.ceil(sigma * 5) // 2 * 2 + 1
    size = int(w) // 2 # floor division eg. 5 // 2 = 2.5 -> 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))*normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    Ix = signal.convolve2d(img, Kx, boundary='fill', mode='same')
    Iy = signal.convolve2d(img, Ky, boundary='fill', mode='same')

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255.0
    theta = np.arctan2(Iy, Ix)
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
# def threshold(img, lowThresholdRatio=0.3, highThresholdRatio=0.6):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def coefficient(x,y):
    # https://stackoverflow.com/questions/19175037/determine-a-b-c-of-quadratic-equation-using-data-points
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    a = ((y1 / ((x1 - x2) * (x1 - x3)))
        + (y2 / ((x2 - x1) * (x2 - x3)))
        + (y3 / ((x3 - x1) * (x3 - x2))))

    b = (-y1 * (x2 + x3) / ((x1 - x2) * (x1 - x3))
         -y2 * (x1 + x3) / ((x2 - x1) * (x2 - x3))
         -y3 * (x1 + x2) / ((x3 - x1) * (x3 - x2)))

    c = (y1 * x2 * x3 / ((x1 - x2) * (x1 - x3))
        + y2 * x1 * x3 / ((x2 - x1) * (x2 - x3))
        + y3 * x1 * x2 / ((x3 - x1) * (x3 - x2)))

    return a, b, c

def find_true_edge(img):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            # x direction
            x = [img[i - 1, j], img[i, j], img[i + 1, j]]
            y = [0, 0, 0]
            a, b, c = coefficient(x, y)
            x_0 = round(-b / (2 * a))

            # y direction
            x = [img[i, j - 1], img[i, j], img[i, j + 1]]
            y = [0, 0, 0]
            a, b, c = coefficient(x, y)
            y_0 = round(-b / (2 * a))
    return result
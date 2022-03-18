#matplotlib inline
from utils import utils
import canny_edge_detector as ced

#Load Image and convert to Greyscale
imgs = utils.load_data()
utils.visualize(imgs, 'gray')

#Canny Edge Detector
detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
imgs_final = detector.detect()

#Final result
utils.visualize(imgs_final, 'gray')
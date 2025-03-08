import cv2
import numpy as np
from skimage.color import rgb2gray
from scipy import signal
import time
from PyQt5.QtGui import QImage

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def myHarris(image, factor):
    '''
    Compute Harris operator using hessian matrix of the image 
    input : image 
    output : Harris operator.'''
    start_time = time.time()  # Start time measurement
    # Convert RGB image to grayscale
    if len(image.shape) > 2:
        image_gray = rgb2gray(image)
    else:
        image_gray = image

    # x derivative
    sobelx = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # y derivative
    sobely = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    # Get Ixx 
    # To get second derivative differentiate twice.
    Ixx = signal.convolve2d(signal.convolve2d(image_gray, sobelx, "same"), sobelx, "same")     

    # Iyy  
    Iyy = signal.convolve2d(signal.convolve2d(image_gray, sobely, "same"), sobely, "same")

    # Ixy Image 
    Ixy = signal.convolve2d(signal.convolve2d(image_gray, sobelx, "same"), sobely, "same")

    # Get Determinant and trace 
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy

    # Harris is det(H) - a * trace(H) let a = 0.2 
    H = det - 0.2 * trace
    
    # Threshold for corner detection
    threshold = 0.2 / factor * np.max(H)

    # Find Harris corners
    corner_coords = np.argwhere(H > threshold)

    # Draw corners on the image
    for y, x in corner_coords:
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Draw a red circle at each corner


    end_time = time.time()  # End time measurement
    computation_time = end_time - start_time  # Compute computation 
    
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImg, computation_time

   
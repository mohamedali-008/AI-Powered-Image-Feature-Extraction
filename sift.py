import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
from PyQt5.QtGui import QImage
def gaussian_kernel(size,k, sigma):
    """
    Generate a Gaussian kernel.
    """
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*(sigma*k)**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*(sigma*k)**2)), (size, size))
    return kernel / np.sum(kernel)

def convolve(image, kernel, stride=1):
    """
    Perform 2D convolution between image and kernel with zero-padding and stride.
    """
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    padded_image = np.pad(image, ((padding, padding), (padding, padding)))
    output_size_0 = ((padded_image.shape[0] - kernel_size) // stride) + 1
    output_size_1 = ((padded_image.shape[1] - kernel_size) // stride) + 1
    result = np.zeros((output_size_0, output_size_1), dtype=image.dtype)
    for i in range(0, padded_image.shape[0] - kernel_size + 1, stride):
        for j in range(0, padded_image.shape[1] - kernel_size + 1, stride):
            result[i//stride, j//stride] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    return result

def detect_keypoints(image, num_octaves=1, num_scales=5, sigma=1.6, contrast_threshold=6):
    """
    Detect keypoints using Difference of Gaussians (DoG) and scale space extrema detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = []
    for octave in range(num_octaves):
        octave_images = []
        # k = 2**(1/num_scales)  # Scale factor between adjacent scales
       
        # Generate octave images
        for i in range(num_scales):
            k = 1.4 ** i
            octave_images.append(convolve(gray, gaussian_kernel(5,k, sigma)))
        
        last_dog = None
        next_dog = None
        for i in range(0, len(octave_images) - 3):
            prev_dog = octave_images[i+1] - octave_images[i]
            curr_dog = octave_images[i + 2] - octave_images[i+1]
            next_dog = octave_images[i+3] - octave_images[i+2]
            # Find extrema in DoG space
            for x in range(0, prev_dog.shape[0]-2 ):
                for y in range(0, prev_dog.shape[1]-2 ):
                    prev_patch = prev_dog[x:x+3, y:y+3]
                    curr_patch = curr_dog[x:x+3, y:y+3]
                    next_patch = next_dog[x:x+3, y:y+3]
                    extremes = {
                        'prev':(np.argmax(prev_patch),np.argmin(prev_patch)),
                        'curr':(np.argmax(curr_patch),np.argmin(curr_patch)),
                        'next':(np.argmax(next_patch),np.argmin(next_patch)),
                    } 
                    if extremes['curr'][0]>extremes['next'][0] and extremes['curr'][0]>extremes['prev'][0] and extremes['curr'][0] >= contrast_threshold:
                        i,j = np.unravel_index(extremes['curr'][0], curr_patch.shape)
                        if i == curr_patch.shape[0]//2 and j == curr_patch.shape[1]//2 :
                            keypoints.append((i+x, j+y))
                    elif extremes['curr'][1]<extremes['next'][1] and extremes['curr'][1]<extremes['prev'][1] and extremes['curr'][0] >=contrast_threshold :
                        i,j = np.unravel_index(extremes['curr'][1], curr_patch.shape)
                        if i == curr_patch.shape[0]//2 and j == curr_patch.shape[1]//2 :
                            keypoints.append((i+x, j+y))
            
    return np.array(keypoints)

def compute_magnitude_direction(image):
    """
    Compute gradient magnitude and direction.
    """
    prewit_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    prewit_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    gradient_x = convolve(image,prewit_x )
    gradient_y = convolve(image,prewit_y)
    gradient_magnitude = np.sqrt(gradient_x**2 +gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180/np.pi
    return gradient_magnitude, gradient_direction

def assign_orientation(image, keypoints):
    """
    Assign orientations to keypoints.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_magnitude, gradient_direction = compute_magnitude_direction(image)
    # quantize the gradient direction to 36 bins
    quantized_angles = np.round(gradient_direction, decimals=-1).astype(np.int16)
    assigned_keypoints = []
    for point in keypoints:
        x = point[0]
        y = point[1]
        window = quantized_angles[x-1:x+2,y-1:y+2] 
        most_frequent = np.bincount(window.flatten())
        dominant_direction = np.argmax(most_frequent)
        window -= dominant_direction
        window[window < 0] += 360
        quantized_angles[x-1:x+2,y-1:y+2] =window
        assigned_keypoints.append((x,y,dominant_direction))
    return np.array(assigned_keypoints),gradient_magnitude, quantized_angles

def extract_descriptors(assigned_keypoints,gradient_magnitude, gradient_direction):
    # quantize the gradient direction to 8 bins
    bins = np.array([0,45,90,135,180,225,270,315])
    quantized_angles_indices = np.digitize(gradient_direction, bins)
    descriptors= []
    for point in assigned_keypoints:
        x = point[0] 
        y = point[1]
        if x < 8 or y < 8 or x > quantized_angles_indices.shape[1]-9 or y > quantized_angles_indices.shape[0]-9:
            continue
        outer_window = quantized_angles_indices[x-8:x+8,y-8:y+8]
        descriptor = []
        for i in range(0, 4):
            for j in range(0, 4):
                inner_window = outer_window[4*i:4*(i+1),4*j:4*(j+1)]
                # flatten_window = inner_window.flatten()
                # angles = np.array([bins[k-1] for k in flatten_window])
                histogram = np.zeros_like(bins)
                for k in range(0, inner_window.shape[0]):
                    for l in range(0, inner_window.shape[1]):
                        index = inner_window[k, l]-1
                        angle = bins[index]
                        loc = (x-8+j+l,y-8+i+k)
                        histogram[index] += gradient_magnitude[loc[0],loc[1]]
                # for a in angles:
                #     indices = np.where(bins == a)
                #     for index in indices:
                #         histogram[index] += 1
                descriptor.append(histogram)
                
        descriptor = np.array(descriptor).flatten()/np.linalg.norm(np.array(descriptor).flatten())
        descriptors.append((point,descriptor))
    return descriptors

# Read an image
def generate_image(path):
    start_time = time.time()
    image1 = cv2.imread(path)

    angle = 0

    # Get the height and width of the image
    height, width = image1.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Apply the rotation to the image
    image2 = cv2.warpAffine(image1, rotation_matrix, (width, height))

    figure, ax = plt.subplots(1, 2, figsize=(16, 8))


    # # Detect keypoints
    keypoints = detect_keypoints(image1,1,5,1.6,4)
    assigned_keypoints,magnitude, directions = assign_orientation(image1,keypoints)
    descriptors1 = extract_descriptors(assigned_keypoints,magnitude, directions)

    # # Detect keypoints
    keypoints = detect_keypoints(image2,1,5,1.6,4)
    assigned_keypoints,magnitude,  directions = assign_orientation(image2,keypoints)
    descriptors2 = extract_descriptors(assigned_keypoints,magnitude, directions)

    matched_points = []
    for vect1 in descriptors1:
        for vect2 in descriptors2:
            dist = np.linalg.norm(vect1[1]-vect2[1])
            if dist < 0.2:
                matched_points.append((vect1[0],vect2[0]))
                
    # # Draw keypoints on the image
    for pair in matched_points:
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        rgb_color = (red, green, blue)
        cv2.circle(image1, (int(pair[0][0]), int(pair[0][1])), 3, rgb_color, -1)
        cv2.circle(image2, (int(pair[1][0]), int(pair[1][1])), 3, rgb_color, -1)


    # Resize images to have the same height (optional)
    height = min(image1.shape[0], image2.shape[0])
    image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
    image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))
    end_time = time.time()  
    computation_time = end_time - start_time
    # Create a new image by concatenating the two images horizontally
    concatenated_image = cv2.hconcat([image1, image2])
    height, width, channel = concatenated_image.shape
    bytesPerLine = 3 * width
    qImage = QImage(concatenated_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImage,computation_time
   
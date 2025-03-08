import cv2
import numpy as np
import time
from PyQt5.QtGui import QImage

class App:
    def __init__(self):
        pass

    def calculateSimilarity(self, original, template, x, y, similarity_measure):
        roi = original[y:y+template.shape[0], x:x+template.shape[1]]

        if similarity_measure == 'NCC':
            # Calculate means
            mean1 = np.mean(roi)
            mean2 = np.mean(template)

            # Calculate cross-correlation
            cross_corr = np.sum((roi - mean1) * (template - mean2))

            # Calculate standard deviations
            std_dev1 = np.std(roi)
            std_dev2 = np.std(template)

            # Calculate NCC
            similarity_score = cross_corr / (std_dev1 * std_dev2)
        elif similarity_measure == 'SSD':
            # Calculate SSD
            diff = roi - template
            ssd = np.sum(diff ** 2)
            similarity_score = ssd

        return similarity_score

    def findTemplate(self, original, template, similarity_measure,threshold):
        start_time = time.time()
        max_similarity_score = float('-inf')
        best_match_x = 0
        best_match_y = 0

        # Iterate over each pixel in the original image
        for y in range(original.shape[0] - template.shape[0] + 1):
            for x in range(original.shape[1] - template.shape[1] + 1):
                # Calculate similarity score between original image and template
                similarity_score = self.calculateSimilarity(original, template, x, y, similarity_measure)

                # Update maximum similarity score and position
                if similarity_score > max_similarity_score:
                    max_similarity_score = similarity_score
                    best_match_x = x
                    best_match_y = y

        # Draw rectangle around matched region
        result = original.copy()
        # cv2.rectangle(result, (best_match_x, best_match_y), (best_match_x + template.shape[1], best_match_y + template.shape[0]), (255, 0, 0), 2)

        # Find key points and descriptors using SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(original, None)
        kp2, des2 = sift.detectAndCompute(template, None)

        # Match descriptors using BFMatcher with k=2 (return top 2 matches)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < float(threshold) * n.distance:
                good_matches.append(m)
        end_time = time.time()  # End time measurement
        computation_time = end_time - start_time  # Compute computation        
                

        # Draw only good matches
        result_with_lines = cv2.drawMatches(original, kp1, template, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        height, width, channel = result_with_lines.shape
        bytesPerLine = 3 * width
        qImg = QImage(result_with_lines.data, width, height, bytesPerLine, QImage.Format_RGB888)
        

        return qImg,computation_time

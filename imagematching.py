import cv2
import numpy as np
import time
from PyQt5.QtGui import QImage

class App:
    def __init__(self):
        pass

    def calculateSimilarity(self, des1, des2, similarity_measure):
        if similarity_measure == 'NCC':
            # Calculate means
            mean1 = np.mean(des1)
            mean2 = np.mean(des2)

            # Calculate cross-correlation
            cross_corr = np.sum((des1 - mean1) * (des2 - mean2))

            # Calculate standard deviations
            std_dev1 = np.std(des1)
            std_dev2 = np.std(des2)

            # Calculate NCC
            similarity_score = (cross_corr / (std_dev1 * std_dev2)) * 1 / (len(des2) - 1)
        elif similarity_measure == 'SSD':
            # Calculate SSD
            diff = des1 - des2
            ssd = np.sum(diff ** 2)
            similarity_score = np.sqrt(ssd)

        return similarity_score

    def findTemplate(self, original, template, similarity_measure, threshold):
        start_time = time.time()   
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(original, None)
        kp2, des2 = sift.detectAndCompute(template, None)

        best_matches = []  
        for i in range(len(des1)):
            similarity_scores = []  
            # Calculate similarity score between des1[i] and all descriptors in des2
            for j in range(len(des2)):
                similarity_scores.append(self.calculateSimilarity(des1[i], des2[j], similarity_measure))
            # Find the index of the descriptor in des2 with the highest similarity score
            best_match_index = np.argmax(similarity_scores)
            best_matches.append(cv2.DMatch(i, best_match_index, similarity_scores[best_match_index]))

        # Filter matches based on the threshold
        good_matches = []
        for match in best_matches:
            print(match.distance)
            if match.distance < float(threshold):
                good_matches.append(match)

        end_time = time.time()  # End time measurement
        computation_time = end_time - start_time  # Compute computation time

        # Draw only good matches
        result_with_lines = cv2.drawMatches(original, kp1, template, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        height, width, channel = result_with_lines.shape
        bytesPerLine = 3 * width
        qImg = QImage(result_with_lines.data, width, height, bytesPerLine, QImage.Format_RGB888)

        return qImg, computation_time

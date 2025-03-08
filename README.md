# MediVision: AI-Powered Image Feature Extraction

## Overview
MediVision is an **AI-driven computer vision application** designed for **feature detection, template matching, and keypoint-based image processing**. It is specifically optimized for **medical imaging, research, and healthcare applications**, making it ideal for tasks such as **medical image registration, object recognition, and anomaly detection**.

## Features

- **Harris Corner Detection**: Detects key features in medical scans using gradient-based calculations.
- **Feature Matching**: Uses **SSD (Sum of Squared Differences) and NCC (Normalized Cross-Correlation)** for precise template matching.
- **SIFT Keypoint Detection**: Extracts scale-invariant keypoints for detailed medical image analysis.
- **Gaussian Filtering**: Applies noise reduction for enhanced image clarity.
- **Performance Evaluation**: Measures computation time and effect of parameter tuning for accuracy.

## Feature Detection & Matching Techniques

### 1. **Harris Corner Detection**
- Computes image derivatives using **Sobel filters**.
- Generates corner response function **H** and applies a threshold.
- Detects corners and visualizes them with **red markers**.

**Example Output:**

![Harris Corner Detection](https://github.com/MO-Nigo/AI-Powered-Image-Feature-Extraction/blob/main/Images/Screenshot%202025-03-08%20073227.png)

### 2. **Feature Matching (SSD & NCC)**
- Extracts SIFT descriptors from medical images.
- Computes similarity using:
  - **SSD (Sum of Squared Differences)** → Faster but less precise.
  - **NCC (Normalized Cross-Correlation)** → More accurate for precise medical feature matching.
- Matches features and visualizes connections.

**Example Outputs:**

- **SSD (Threshold = 630)**  
  ![SSD Feature Matching](https://github.com/MO-Nigo/AI-Powered-Image-Feature-Extraction/blob/main/Images/Screenshot%202025-03-08%20074254.png)
- **NCC (Threshold = 0.5)**  
  ![NCC Feature Matching](images/ncc_05.png)

### 3. **SIFT Keypoint Detection & Descriptor Extraction**
- Generates a **Gaussian pyramid** and applies **Difference of Gaussians (DoG)**.
- Computes **gradient magnitude and orientation**.
- Extracts and visualizes keypoints for high-detail medical image processing.

**Example Output:**

![SIFT Keypoint Detection](images/sift_keypoints.png)

## Installation
To set up and run the application, install the dependencies:

```bash
pip install opencv-python numpy matplotlib
```

## Usage
Run the feature detection and matching scripts:

```bash
python harris_detection.py --input images/sample.png --output images/harris_output.png
```

```bash
python feature_matching.py --input1 images/img1.png --input2 images/img2.png --output images/match_output.png
```

```bash
python sift_detection.py --input images/sample.png --output images/sift_output.png
```

## Future Enhancements
- **Optimize performance** for real-time medical image processing.
- **Implement deep learning-based feature extraction** for anomaly detection.
- **Extend dataset to improve feature matching accuracy in various medical imaging modalities.**

## License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.


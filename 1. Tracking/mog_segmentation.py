"""
    Trying to use Mixture Of Gaussians + connectedComponentsWithStats
    https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
    
    Connected components is basically blob extraction, region labeling. Used for segmentation
"""

import cv2
import numpy as np
import mediapipe

# Initialize MediaPipe Hands module
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mediapipe.solutions.drawing_utils

video = cv2.VideoCapture("videos/video.mp4")
N_GAUSS = 5
BACKGROUND_THRESHOLD = 0.8
NOISE_SIGMA = 1
HISTORY = 100  # t
ALPHA = 0.1
mog_subtractor = cv2.createBackgroundSubtractorMOG2()
min_area_threshold = 1300
max_area_threshold = 3000
# mog_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_GAUSS, BACKGROUND_THRESHOLD, NOISE_SIGMA)
# knn_subtractor = cv2.createBackgroundSubtractorKNN(history= HISTORY)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # preprocess the image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # Applying threshold
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[
        1
    ]

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to
    # store all the output components
    output = np.zeros(gray_img.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):

        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]

        if (area > min_area_threshold) and (area < max_area_threshold):
            # Create a new image for bounding boxes
            new_img = frame.copy()

            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]

            # Coordinate of the bounding box
            pt1 = (x1, y1)
            pt2 = (x1 + w, y1 + h)
            (X, Y) = centroid[i]

            # Bounding boxes for each component
            cv2.rectangle(new_img, pt1, pt2, (0, 255, 0), 3)
            cv2.circle(new_img, (int(X), int(Y)), 4, (0, 0, 255), -1)

            # Create a new array to show individual component
            component = np.zeros(gray_img.shape, dtype="uint8")
            componentMask = (label_ids == i).astype("uint8") * 255

            # Apply the mask using the bitwise operator
            component = cv2.bitwise_or(component, componentMask)
            output = cv2.bitwise_or(output, componentMask)

            # Show the final images
            cv2.imshow("Image", new_img)
            cv2.imshow("Individual Component", component)
            cv2.imshow("Filtered Components", output)
            cv2.waitKey(1)

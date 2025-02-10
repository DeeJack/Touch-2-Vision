"""
    Just MoG
"""

import cv2
import numpy as np
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Initialize MediaPipe Hands module
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mediapipe.solutions.drawing_utils

video = cv2.VideoCapture("videos/video.mp4")
outputWriter = cv2.VideoWriter(
    "results/mog2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (int(video.get(3)), int(video.get(4))),
    True,
)
N_GAUSS = 5
BACKGROUND_THRESHOLD = 0.8
NOISE_SIGMA = 1
HISTORY = 100  # t
ALPHA = 0.5
mog_subtractor = cv2.createBackgroundSubtractorMOG2()
min_area_threshold = 1100
max_area_threshold = 3000
# mog_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_GAUSS, BACKGROUND_THRESHOLD, NOISE_SIGMA)
# knn_subtractor = cv2.createBackgroundSubtractorKNN(history= HISTORY)

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    foreground_mask = mog_subtractor.apply(frame_greyscale, learningRate=ALPHA)
    background = mog_subtractor.getBackgroundImage()
    # cv2.imshow('frame', frame)
    cv2.imshow("foreground", foreground_mask)
    # cv2.imshow('background', cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

    _, thresh = cv2.threshold(foreground_mask, 110, 200, cv2.THRESH_BINARY)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)

    # Filter contours based on area and other properties
    # contours = [
    #     contour
    #     for contour in contours
    #     if cv2.contourArea(contour) > min_area_threshold
    #     and cv2.contourArea(contour) < max_area_threshold
    # ]

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Foreground Segmentation", frame)
    cv2.imshow("Threshold", thresh)
    outputWriter.write(frame)

    cv2.waitKey(1)

video.release()
outputWriter.release()
cv2.destroyAllWindows()

"""
    Mediapipe, from video to imgs
"""

import cv2
import numpy as np
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Initialize MediaPipe Hands module
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1)

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mediapipe.solutions.drawing_utils

video = cv2.VideoCapture('videos/video.mp4')
# save_name = "results/hand_" + "video.mp4"
# Ensure the output video is in color by setting the last parameter to True
# outputWriter = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(video.get(3)), int(video.get(4))), True)
N_GAUSS = 5
BACKGROUND_THRESHOLD = 0.8
NOISE_SIGMA = 1
HISTORY = 100 # t
ALPHA = 0.1
mog_subtractor = cv2.createBackgroundSubtractorMOG2()
min_area_threshold = 500
max_area_threshold = 1500
# mog_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_GAUSS, BACKGROUND_THRESHOLD, NOISE_SIGMA)
# knn_subtractor = cv2.createBackgroundSubtractorKNN(history= HISTORY)
count = 0

while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 50, (255, 255, 255), -1)
            # Draw landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)
    # cv2.imshow('asd', frame_rgb)
    
    # outputWriter.write(frame)
    
    # cv2.imwrite(f"masks/frame_{count:0000}.png", mask)
    
    count += 1
    cv2.waitKey(1)
# outputWriter.release()
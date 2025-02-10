"""
    Use mediapipe, read and save to individual frames, instead of creating a video.
"""

import cv2 as cv
import numpy as np
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Initialize MediaPipe Hands module
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1
)

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mediapipe.solutions.drawing_utils

frames_dir = "../frames/"
count = 0

while True:
    frame = cv.imread(f"{frames_dir}/frame_{count}.jpg")

    if frame is None:
        break

    # Convert the frame to RGB format
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        left = 999999
        right = 0
        top = 999999
        bottom = 0
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                left = min(left, landmark.x)
                right = max(right, landmark.x)
                top = min(top, landmark.y)
                bottom = max(bottom, landmark.y)
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            left, right = int(left * frame.shape[1]), int(right * frame.shape[1])
            top, bottom = int(top * frame.shape[0]), int(bottom * frame.shape[0])
            left, right, top, bottom = (
                max(0, left - 50),
                min(frame.shape[1], right + 50),
                max(0, top - 50),
                min(frame.shape[0], bottom + 50),
            )
            # left, right, top, bottom = max(0, left + 50), min(frame.shape[1], right - 50), max(0, top + 50), min(frame.shape[0], bottom - 50)

            # print(left, right, top, bottom)

            # Hide the hand region by drawing a rectangle over it
            cv.rectangle(mask, (left, bottom), (right, top), (255, 255, 255), -1)

    cv.imwrite(f"../masks/frame_{count:0000}.png", mask)

    count += 1
    cv.waitKey(1)

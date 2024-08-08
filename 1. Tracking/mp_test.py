"""
    Test MediaPipe for hand detection, use boundaries to create a rectangle
"""

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# For video input:
cap = cv2.VideoCapture("videos/video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box coordinates of the hand
            x_min, y_min = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]
            ), int(
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0]
            )
            x_max, y_max = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                * frame.shape[1]
            ), int(
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                * frame.shape[0]
            )

            # Hide the hand region by drawing a rectangle over it
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

    # Display the modified frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

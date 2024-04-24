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
save_name = "results/hand_" + "video.mp4"
outputWriter = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(video.get(3)), int(video.get(4))), False)
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

while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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
            left, right, top, bottom = max(0, left - 50), min(frame.shape[1], right + 50), max(0, top - 50), min(frame.shape[0], bottom + 50)
            # print(left, right, top, bottom)
            
            # Hide the hand region by drawing a rectangle over it
            cv2.rectangle(mask, (left, bottom), (right, top), (255, 255, 255), -1)
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         for landmark in hand_landmarks.landmark:
    #             cv2.circle(mask, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 50, (255, 255, 255), -1)
            # Draw landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)
    cv2.imshow('asd', mask)
    
    outputWriter.write(mask)
    
    # frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # foreground_mask = mog_subtractor.apply(frame_greyscale, learningRate=ALPHA)
    # background = mog_subtractor.getBackgroundImage()
    # cv2.imshow('frame', frame)
    # cv2.imshow('foreground', mask)
    # cv2.imshow('background', cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    
    # Find contours in the foreground mask
    # contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    
    # Filter contours based on area and other properties
    # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area_threshold and cv2.contourArea(contour) < max_area_threshold]

    # # Draw contours on the original frame
    # cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 2)

    # # Display the result
    # cv2.imshow('Foreground Segmentation', frame)
    cv2.waitKey(1)
outputWriter.release()
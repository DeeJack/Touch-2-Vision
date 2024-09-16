import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.25)

def create_hand_and_sensor_mask(frame, results):
    hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(hand_mask, (x, y), 10, 255, -1)

            # Get the positions of the index finger tip and thumb tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])

            # Estimate the position of the sensor (e.g., below the palm, near the index finger and thumb)
            sensor_x = (index_x + thumb_x) // 2
            sensor_y = (index_y + thumb_y) // 2 + 20  # Adjust offset as needed

            # Create a mask for the sensor
            cv2.circle(hand_mask, (sensor_x, sensor_y), 30, 255, -1)
    
    return hand_mask

def extract_frames_and_masks(video_path, frames_dir, masks_dir):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        mask = create_hand_and_sensor_mask(frame, results)
        frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        mask_path = os.path.join(masks_dir, f'mask_{frame_count:04d}.png')
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(mask_path, mask)
        frame_count += 1
    cap.release()

video_path = 'videos/video.mp4'
frames_dir = 'frames'
masks_dir = 'masks'
extract_frames_and_masks(video_path, frames_dir, masks_dir)

# Create the video with only the masks
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
mask_video = cv2.VideoWriter('mask_video.mp4', fourcc, 30, (640, 480))

for mask_filename in os.listdir(masks_dir):
    mask_path = os.path.join(masks_dir, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_video.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

mask_video.release()
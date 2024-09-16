import cv2
import numpy as np
import os
import cvlib as cv
import mediapipe as mp
from cvlib.object_detection import draw_bbox

# Paths to YOLO configuration and weights files
# config_path = 'yolov4-tiny.cfg'
# weights_path = 'yolov4-tiny.weights'
# class_names_path = 'coco.names'

# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)

# # Load class names
# with open(class_names_path, 'r') as f:
#     class_names = f.read().strip().split('\n')

def stabilize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32)
    
    # Read the first frame
    ret, prev = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        return
    
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    for i in range(n_frames-1):
        # Read next frame
        ret, curr = cap.read()
        if not ret:
            print(f"Failed to read frame {i+1}.")
            break
        
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Ensure we have enough points
        if len(prev_pts) < 4 or len(curr_pts) < 4:
            # Skip this frame, use identity transform
            dx, dy, da = 0, 0, 0
        else:
            # Find transformation matrix
            m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
            if m is None:
                # Skip this frame, use identity transform
                dx, dy, da = 0, 0, 0
            else:
                # Extract translation
                dx = m[0, 2]
                dy = m[1, 2]
                
                # Extract rotation angle
                da = np.arctan2(m[1, 0], m[0, 0])
        
        # Store transformation
        transforms[i] = [dx, dy, da]
        
        # Move to next frame
        prev_gray = curr_gray
    
    # Compute trajectory
    trajectory = np.cumsum(transforms, axis=0)
    
    # Smooth trajectory using moving average filter
    smoothed_trajectory = smooth(trajectory)
    
    # Calculate difference in trajectory
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference
    
    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Write n_frames-1 transformed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for i in range(n_frames-1):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {i}.")
            break
        
        dx, dy, da = transforms_smooth[i]
        
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        
        out.write(frame_stabilized)
    
    cap.release()
    out.release()

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    window_size = 30
    for i in range(3):
        smoothed_trajectory[:, i] = np.convolve(trajectory[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed_trajectory

if not os.path.exists('stabilized_video.mp4'):
    input_path = 'videos/video.mp4'
    output_path = 'stabilized_video.mp4'
    stabilize_video(input_path, output_path)

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_frames_with_hands(input_path, output_dir):
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            frame_path = f"{output_dir}/frame_{frame_number:04d}.png"
            cv2.imwrite(frame_path, frame)
        
        frame_number += 1
    
    cap.release()

# Update paths according to your setup
input_path = 'videos/video.mp4'
output_dir = 'frames_with_hands'
extract_frames_with_hands(input_path, output_dir)

# Close the Mediapipe hand detector
hands.close()

# for frame_filename in os.listdir(input_dir):
#     frame_path = os.path.join(input_dir, frame_filename)
#     output_path = os.path.join(output_dir, frame_filename)
#     inpaint_frame(frame_path, output_path)

# def reintegrate_frames(original_video_path, edited_frames_dir, output_video_path):
#     cap = cv2.VideoCapture(original_video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    
#     frame_number = 0
#     edited_frames = {int(f.split('_')[1].split('.')[0]): f for f in os.listdir(edited_frames_dir)}
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_number in edited_frames:
#             edited_frame_path = os.path.join(edited_frames_dir, edited_frames[frame_number])
#             edited_frame = cv2.imread(edited_frame_path)
#             out.write(edited_frame)
#         else:
#             out.write(frame)
        
#         frame_number += 1
    
#     cap.release()
#     out.release()

# original_video_path = 'stabilized_video.mp4'
# edited_frames_dir = 'inpainted_frames'
# output_video_path = 'final_video.mp4'
# reintegrate_frames(original_video_path, edited_frames_dir, output_video_path)

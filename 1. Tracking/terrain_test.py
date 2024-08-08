import cv2
import numpy as np

"""
    Using Lucas-Kanade optical flow / goodFeaturesToTrack, not too bad.
"""

# Load video
video = cv2.VideoCapture('videos/video.mp4')

# Parameters for feature detection and tracking
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
prev_frame = None
tracked_features = []

# Process frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracked_features = np.array(tracked_features, dtype=np.float32).reshape(-1, 1, 2)
    # Feature detection and tracking
    if prev_frame is not None:
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray, tracked_features, None, **lk_params)
        tracked_features = new_features[status == 1]
        

    # Feature detection
    if prev_frame is None or len(tracked_features) < 10:
        tracked_features = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)[0]
    
    # Draw tracked features
    for x, y in tracked_features:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(30)

    prev_frame = gray.copy()

video.release()
cv2.destroyAllWindows()

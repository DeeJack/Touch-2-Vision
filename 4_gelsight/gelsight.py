"""
    Detect at which frames the user pressed on the GelSight sensor in the video,
    based on Optical Flow, the magnitude of the flow, and a threshold value.
"""

import cv2
import numpy as np

def detect_press(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read video frames.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Compute the magnitude and angle of the flow
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate the mean magnitude
        mean_motion = np.mean(mag)

        # Check if motion exceeds the threshold
        if mean_motion > threshold:
            motion_detected = True
            print("Press detected at frame:", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # Display the frame for visualization
        cv2.imshow("Frame", frame)
        cv2.imshow("Optical Flow Magnitude", cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

    if motion_detected:
        print("Press detection completed.")
    else:
        print("No press detected in the video.")

# Example usage
video_path = "videos/20220607_133934/gelsight.mp4"  # Replace with your video file path
detect_press(video_path, threshold=0.5)

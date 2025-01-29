"""
    Get profile by using dense optical flow and a threshold
"""

import cv2
import numpy as np


def process_gelsight_video_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    object_profiles = []
    previous_frame_gray = None
    count = 0

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame_gray is None:
            previous_frame_gray = current_frame_gray
            continue

        # 1. Calculate Dense Optical Flow using Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(
            previous_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 2. Calculate the magnitude of the flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 3. Threshold the magnitude image to identify areas of significant motion
        _, thresh = cv2.threshold(magnitude, 1.5, 255, cv2.THRESH_BINARY)

        # 4. Optional: Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

        object_profiles.append(morphed)

        previous_frame_gray = current_frame_gray
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return object_profiles


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    object_profiles = process_gelsight_video_optical_flow(video_path)

    if object_profiles:
        display_threshold = 200

        for profile in object_profiles:
            _, display_thresh_profile = cv2.threshold(
                profile, display_threshold, 255, cv2.THRESH_BINARY
            )
            cv2.imshow(
                "Object Profile (Optical Flow - Thresholded)", display_thresh_profile
            )
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

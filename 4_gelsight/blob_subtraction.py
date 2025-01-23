"""
    Uses blob detection to create a mask of stuff not to include in the profile
    Uses background subtraction + the mask for the profile
"""

import cv2
import numpy as np


def process_gelsight_video_masking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    processed_frames = []
    relaxed_frame_gray = None

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    detector = cv2.SimpleBlobDetector_create(params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(frame_gray)

        if relaxed_frame_gray is None:
            relaxed_frame_gray = frame_gray.copy()
            processed_frames.append(frame_gray)
            continue

        # 1. Perform Background Subtraction
        frame_diff = cv2.absdiff(relaxed_frame_gray, frame_gray)
        _, thresh = cv2.threshold(frame_diff, 60, 255, cv2.THRESH_BINARY)

        # 2. Create a mask of the detected blobs
        mask = np.zeros_like(thresh)
        if keypoints:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size / 2)  # Approximate radius
                cv2.circle(mask, (x, y), r + 3, 255, -1)  # Draw filled circles

        # 3. Invert the blob mask to keep everything EXCEPT the blobs
        mask_inv = cv2.bitwise_not(mask)

        # 4. Apply the inverted mask to the thresholded difference image
        object_profile = cv2.bitwise_and(thresh, thresh, mask=mask_inv)

        processed_frames.append(object_profile)

        cv2.imshow("Difference", thresh)
        cv2.imshow("Blob Mask", mask)
        cv2.imshow("Object Profile", object_profile)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return processed_frames


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    object_profiles = process_gelsight_video_masking(video_path)

    if object_profiles:
        for i in range(1, len(object_profiles)):
            cv2.imshow("Object Profile", object_profiles[i])
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

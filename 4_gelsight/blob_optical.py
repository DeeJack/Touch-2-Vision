"""
    Uses blob extraction to exclude the circular artifacts, 
    then optical flow to get the object profile
"""

import cv2
import numpy as np


def process_gelsight_video_hybrid(video_path):
    """
    Processes a Gelsight video using a hybrid approach: marker tracking to mask
    out markers, and optical flow on the remaining regions to find the object profile.

    Args:
        video_path: Path to the Gelsight video file.

    Returns:
        A list of object profiles (binary masks).
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    object_profiles = []
    previous_frame_gray = None

    # Setup SimpleBlobDetector parameters (use your tuned parameters)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    detector = cv2.SimpleBlobDetector_create(params)

    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame_gray is None:
            previous_frame_gray = current_frame_gray
            continue

        # 1. Detect Markers
        keypoints = detector.detect(current_frame_gray)

        # 2. Create a Mask of the Marker Regions
        marker_mask = np.zeros_like(current_frame_gray, dtype=np.uint8)
        if keypoints:
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size / 2)
                cv2.circle(marker_mask, (x, y), r + 4, 255, -1)

        # 3. Invert the Marker Mask to focus on non-marker regions
        non_marker_mask = cv2.bitwise_not(marker_mask)

        # 4. Calculate Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            previous_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 5. Calculate Magnitude of Optical Flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 6. Apply the Non-Marker Mask to the Magnitude Image
        masked_magnitude = cv2.bitwise_and(
            magnitude.astype(np.uint8), magnitude.astype(np.uint8), mask=non_marker_mask
        )

        # 7. Threshold the Masked Magnitude to get the Object Profile
        _, thresh = cv2.threshold(masked_magnitude, 1.0, 255, cv2.THRESH_BINARY)

        # 8. Apply Morphological Operations
        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

        object_profiles.append(morphed)

        # Display Intermediate and Final Results
        # cv2.imshow("Marker Mask", marker_mask)
        # cv2.imshow("Non-Marker Mask", non_marker_mask)
        # cv2.imshow("Optical Flow Magnitude", magnitude)
        # cv2.imshow("Masked Magnitude", masked_magnitude)
        # cv2.imshow("Object Profile (Hybrid)", morphed)
        # if cv2.waitKey(50) & 0xFF == ord("q"):
        #     break

        previous_frame_gray = current_frame_gray
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return object_profiles


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    object_profiles = process_gelsight_video_hybrid(video_path)

    if object_profiles:
        # Process each object profile to find and display contours
        for i, profile in enumerate(object_profiles):
            # 1. Find Contours
            contours, hierarchy = cv2.findContours(
                profile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 2. Draw Contours on the Original Frame
            # Load the corresponding original frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
            ret, original_frame = cap.read()
            cap.release()

            # if ret:
            #     contour_frame = original_frame.copy()
            #     cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)  # Draw in green
            #     cv2.imshow(f"Object Profile with Contours (Frame {i+1})", contour_frame)

            # 3. Draw Contours on the Binary Profile
            contour_profile = cv2.cvtColor(
                profile, cv2.COLOR_GRAY2BGR
            )  # Convert to color to draw
            cv2.drawContours(contour_profile, contours, -1, (0, 255, 0), 2)
            cv2.imshow(f"Contours", contour_profile)

            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

        # Access and work with the contours
        # all_contours = []
        # for profile in object_profiles:
        #     contours, hierarchy = cv2.findContours(profile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     all_contours.append(contours)
        # print("Extracted contours:", all_contours)

"""
    Detect at which frames the user pressed on the GelSight sensor in the video,
    based on background subtraction and motion detection.
"""

import cv2
import numpy as np


def detect_press(video_path, threshold=20):
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read video frames.")
        return

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    background = first_gray.copy().astype(
        "float"
    )  # Initialize background as the first frame
    relaxed_frame = first_gray.copy()
    motion_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update the background
        cv2.accumulateWeighted(gray, background, 0.5)

        # Calculate absolute difference between current frame and background
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))

        # Apply threshold to get the foreground mask
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Use morphological operations to remove noise
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # If contours are found, consider it as motion
        if contours:
            motion_detected = True
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            profile = gray_frame - background
            cv2.imshow("profile", profile)
            cv2.waitKey(1000)
            print("Press detected at frame:", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            # Draw the contours in the original frame
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            path = (
                f"pressed_frame/press_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
            )
            print(path)
            cv2.imwrite(path, frame)

        # Display the frame for visualization
        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground Mask", thresh)

        # Exit on 'q' key
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if motion_detected:
        print("Press detection completed.")
    else:
        print("No press detected in the video.")


video_path = "videos/20220607_133934/gelsight.mp4"
detect_press(video_path, threshold=20)

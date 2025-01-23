"""
    Color based: remove black/dark gray colors.
"""

import cv2
import numpy as np


def process_gelsight_video_color_masking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    object_profiles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_dark_gray = np.array([100, 100, 100], dtype=np.uint8)

        # 1. Create a mask for the black and dark gray regions
        mask = cv2.inRange(frame, lower_black, upper_dark_gray)

        # 2. Invert the mask to select the non-black/dark-gray regions
        mask_inv = cv2.bitwise_not(mask)

        # 3. Apply the inverted mask to the original frame
        # Create a black image with the same shape as the frame
        masked_frame = np.zeros_like(frame)
        # Copy the regions from the original frame where the mask_inv is white
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # 4. Convert the masked frame to grayscale
        gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # 5. Threshold the grayscale masked frame to create a binary profile
        _, object_profile = cv2.threshold(gray_masked, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(object_profile, cv2.MORPH_OPEN, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

        object_profiles.append(morphed)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Black/Dark Gray Mask", mask)
        cv2.imshow("Inverted Mask", mask_inv)
        cv2.imshow("Masked Frame", masked_frame)
        cv2.imshow("Grayscale Masked", gray_masked)
        cv2.imshow("Object Profile (Color Masking)", morphed)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return object_profiles


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    object_profiles = process_gelsight_video_color_masking(video_path)

    if object_profiles:
        for i, profile in enumerate(object_profiles):
            contours, _ = cv2.findContours(
                profile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
                ret, original_frame = cap.read()
                cap.release()
                if ret:
                    contour_frame = original_frame.copy()
                    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
                    cv2.imshow(
                        f"Object Profile (Color Masking) with Contours - Frame {i+1}",
                        contour_frame,
                    )

            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

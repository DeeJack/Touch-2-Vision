"""
Interactive HSV color space segmentation tool for image processing.

This script provides a GUI interface with trackbars to tune HSV (Hue, Saturation, Value) 
thresholds for image segmentation. 
"""

import cv2
import numpy as np


def nothing():
    pass


def interactive_tuning(image_path):
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Create a window for the trackbars.
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    # Create trackbars for lower and upper HSV values.
    cv2.createTrackbar("Lower H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("Lower S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("Lower V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("Upper H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

    while True:
        # Read the current trackbar positions.
        lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
        lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
        lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
        upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
        upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
        upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")
        lower_hsv = np.array([lower_h, lower_s, lower_v])
        upper_hsv = np.array([upper_h, upper_s, upper_v])

        # Convert image to HSV and threshold it.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Apply morphological operations to reduce noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_contours = image.copy()
        if contours:
            # Draw all found contours in green.
            cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

        # Display the mask and the contour overlay.
        cv2.imshow("Mask", mask_clean)
        cv2.imshow("Contours", image_contours)

        # Press ESC to exit.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    fps = 27
    frame_num = (8 * 60) * fps + 40  # 8 min, 7 seconds
    target_frame = f"./results/inpaint2.jpg"
    interactive_tuning(target_frame)

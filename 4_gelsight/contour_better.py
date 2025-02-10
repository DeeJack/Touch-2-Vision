"""
    Applies a series of image processing steps to extract the profile of an object in an image.
    Such as noise reduction, contrast enhancement, edge-preserving smoothing, adaptive thresholding,
    Then, uses contour detection and refinement to extract the object profile.
"""

import cv2
import numpy as np


def extract_object_profile(img):
    # Load image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing steps
    # 1. Noise reduction using median filter
    denoised = cv2.medianBlur(gray, 5)

    # 2. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # 3. Edge-preserving smoothing
    blurred = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)

    # 4. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
    )

    # 5. Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Find and select contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img  # Return original if no contours found

    # Filter and select largest contour
    max_contour = max(contours, key=cv2.contourArea)

    # 7. Contour approximation and smoothing
    epsilon = 0.001 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # 8. Create mask and refine edges
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [approx], -1, 255, -1)

    # 9. Final refinement using grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = cv2.boundingRect(approx)
    mask_grabcut = np.where((mask == 255), cv2.GC_FGD, cv2.GC_BGD).astype("uint8")
    cv2.grabCut(img, mask_grabcut, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 255).astype(
        "uint8"
    )

    # Find final contours from refined mask
    final_contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if final_contours:
        max_contour = max(final_contours, key=cv2.contourArea)

    # Draw the final contour on original image
    result = img.copy()
    cv2.drawContours(result, [max_contour], -1, (0, 255, 0), 2)

    return result


inpaint_video = "results/inpaint2.mp4"
cap = cv2.VideoCapture(inpaint_video)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = extract_object_profile(frame)
    frames.append(result_frame)

cap.release()

for i, frame in enumerate(frames):
    cv2.imwrite(f"results/profile/contour_better_{i}.png", frame)
"""
    Similar to the previous one, but this one uses a more advanced preprocessing pipeline to enhance the image quality before contour extraction.
"""

import cv2
import numpy as np


def extract_object_profile_v2(img, area_threshold=(5000, 50000), debug=False):
    # Load image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhanced preprocessing pipeline
    # 1. Multi-step noise reduction
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=7, templateWindowSize=7, searchWindowSize=21
    )

    cv2.imshow("denoised", denoised)
    cv2.waitKey(1)

    # 2. Adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    contrast_enhanced = clahe.apply(denoised)

    cv2.imshow("contrast_enhanced", contrast_enhanced)
    cv2.waitKey(1)

    # 3. Edge-aware smoothing
    blurred = cv2.bilateralFilter(contrast_enhanced, 15, 75, 75)

    cv2.imshow("blurred", blurred)
    cv2.waitKey(1)

    # 4. Multi-level thresholding
    _, th1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    combined_thresh = cv2.bitwise_and(th1, th2)

    cv2.imshow("combined_thresh", combined_thresh)
    cv2.waitKey(1)

    # 5. Morphological refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imshow("morph", morph)
    cv2.waitKey(1)

    # Find contours with area constraints
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and convexity
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)

        if area_threshold[0] < area < area_threshold[1]:
            valid_contours.append(cnt)

    if not valid_contours:
        return img

    # Select largest valid contour
    main_contour = max(valid_contours, key=cv2.contourArea)

    # Contour refinement using convex hull
    hull = cv2.convexHull(main_contour)

    # Create a mask for final verification
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    # Final edge-preserving refinement
    refined = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
    refined_gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)

    # Detect edges on refined image
    edges = cv2.Canny(refined_gray, 30, 100)

    # Combine with initial mask
    final_mask = cv2.bitwise_and(mask, edges)

    # Find final contours
    final_contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if final_contours:
        main_contour = max(final_contours, key=cv2.contourArea)

    # Draw result
    result = img.copy()
    cv2.drawContours(result, [main_contour], -1, (0, 255, 0), 2)

    if debug:
        debug_imgs = {
            "denoised": denoised,
            "contrast_enhanced": contrast_enhanced,
            "combined_thresh": combined_thresh,
            "morph": morph,
            "mask": mask,
            "edges": edges,
            "final_mask": final_mask,
        }
        for name, dbg_img in debug_imgs.items():
            cv2.imshow(name, dbg_img)

    return result


inpaint_video = "videos/20220607_133934/gelsight.mp4"
cap = cv2.VideoCapture(inpaint_video)

frames = []
frame_count = 0
while frame_count < 150:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame_count += 1
    print(frame_count)

    result_frame = extract_object_profile_v2(
        frame,
        area_threshold=(int(0.2 * h * w), int(0.9 * h * w)),  # 90% of image area
        debug=True,
    )
    frames.append(result_frame)

cap.release()

for i, frame in enumerate(frames):
    cv2.imwrite(f"results/profile/contour_better_{i}.png", frame)

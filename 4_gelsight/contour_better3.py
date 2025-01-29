import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_gelsight_profile(img, inpainted_img):
    # Load image in original color
    h, w = img.shape[:2]

    # 1. Analyze in LAB color space - better for texture/deformation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Texture analysis using Local Binary Patterns
    radius = 3  # For surface texture patterns
    n_points = 8 * radius
    lbp = local_binary_pattern(l_channel, n_points, radius, method="uniform")
    lbp = (lbp * 255).astype(np.uint8)

    # 3. Gradient magnitude in pressure-sensitive channels
    sobel_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(b_channel, cv2.CV_64F, 0, 1, ksize=5)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_mag = np.uint8(255 * gradient_mag / gradient_mag.max())

    # 4. Multi-modal fusion
    fused = cv2.addWeighted(lbp, 0.7, gradient_mag, 0.3, 0)

    # 5. Adaptive contrast stretching
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32, 32))
    enhanced = clahe.apply(fused)

    # 6. Morphological Top-Hat for small artifact removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # 7. Dynamic thresholding
    thresh = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 5
    )

    # 8. Geometric constraints
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (0.2 * h * w < area < 0.7 * h * w):  # Reject too small/large
            continue
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate 20% margins
        margin_x = int(0.2 * w)
        margin_y = int(0.2 * h)
        
        # Create a mask for the inner 60% region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # Cut out the margins
        cut_cnt = cnt.copy()
        cut_cnt[:, 0, 0] = np.clip(cut_cnt[:, 0, 0], x + margin_x, x + w - margin_x)
        cut_cnt[:, 0, 1] = np.clip(cut_cnt[:, 0, 1], y + margin_y, y + h - margin_y)
        
        # Only add the cut contour if it's still valid
        cut_area = cv2.contourArea(cut_cnt)
        if cut_area > 0:  # Ensure we still have a valid contour after cutting
            valid_contours.append(cut_cnt)

        # Shape compactness filter
        #perimeter = cv2.arcLength(cnt, True)
        #compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        #if compactness < 0.2:  # Reject linear artifacts
        #    continue

        #valid_contours.append(cnt)

    if not valid_contours:
        return img

    # 9. Convex hull of combined contours
    all_points = np.vstack(valid_contours)
    hull = cv2.convexHull(all_points)

    # 10. Refine using physical deformation constraints
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    # 11. Edge-preserving refinement
    #refined = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
    #result = cv2.bitwise_and(refined, refined, mask=mask)

    # Draw boundary
    cv2.drawContours(inpainted_img, [hull], -1, (0, 255, 0), 2)
    
    cv2.imshow("Contour", inpainted_img)
    cv2.waitKey(1)

    return inpainted_img


# Usage
video = "videos/20220607_133934/gelsight.mp4"
inpainted_video = "results/inpaint2.mp4"
cap = cv2.VideoCapture(video)
cap2 = cv2.VideoCapture(inpainted_video)

frames = []
frame_count = 0
while frame_count < 150:
    ret, frame = cap.read()
    ret, frame2 = cap2.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame_count += 1
    print(frame_count)

    result_frame = extract_gelsight_profile(
        frame, frame2
    )
    frames.append(result_frame)

cap.release()

for i, frame in enumerate(frames):
    cv2.imwrite(f"results/profile/contour_better_{i}.png", frame)

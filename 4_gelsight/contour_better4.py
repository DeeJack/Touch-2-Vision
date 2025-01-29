import cv2
import numpy as np

def extract_object_profile(image):
    # Load the image

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Enhance color channels using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    a_enhanced = clahe.apply(a)
    b_enhanced = clahe.apply(b)

    # Compute gradient magnitudes for both color channels
    def compute_gradient(channel):
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.normalize(np.sqrt(grad_x**2 + grad_y**2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    gradient_a = compute_gradient(a_enhanced)
    gradient_b = compute_gradient(b_enhanced)

    # Combine gradients and threshold
    combined = cv2.addWeighted(gradient_a, 0.5, gradient_b, 0.5, 0)
    _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find and select largest contour
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    # Find and select largest contour
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    # Calculate image area and valid contour size range
    image_area = image.shape[0] * image.shape[1]
    min_area = 0.2 * image_area  # 30% of image area
    max_area = 0.9 * image_area  # 80% of image area
    
    # Filter contours by area
    valid_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    
    if not valid_contours:
        return image
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Smooth and approximate contour
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Draw result
    result = image.copy()
    cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
    
    cv2.imshow("asd", result)
    cv2.waitKey(1)
    
    return result

# Example usage
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

    result_frame = extract_object_profile(
        frame,
    )
    frames.append(result_frame)

cap.release()

#for i, frame in enumerate(frames):
    #cv2.imwrite(f"results/profile/contour_better_{i}.png", frame)


# Edge-aware mask refinement
    edges = cv2.Canny(enhanced_lab, 50, 150)
    mask = mask.astype(np.uint8) * 255  # Convert boolean mask to uint8
    mask = cv2.bitwise_and(mask, edges)
    
    # Find and validate contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contour validation criteria
    valid_contours = []
    img_area = img.shape[0] * img.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        ## 1. Area threshold (5-40% of image area)
        #if not (0.05*img_area < area < 0.4*img_area):
        #    continue
        #    
        ## 2. Aspect ratio (0.3-3.0)
        #aspect_ratio = w / h
        #if not (0.3 < aspect_ratio < 3.0):
        #    continue
        #    
        ## 3. Solidity (contour area vs convex hull area)
        #hull = cv2.convexHull(cnt)
        #hull_area = cv2.contourArea(hull)
        #if hull_area == 0:
        #    continue
        #solidity = float(area)/hull_area
        #if solidity < 0.85:
        #    continue
        #    
        valid_contours.append(cnt)
    
    if not valid_contours:
        return img, None
    
    # Select best contour using weighted score
    best_score = -np.inf
    best_contour = None
    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Score components
        area_score = min(area/(0.3*img_area), 1.0)  # Prefer medium-sized
        aspect_score = 1 - abs(1 - (w/h))  # Closer to 1:1 gets higher score
        #solidity_score = solidity
        
        total_score = 0.5*area_score + 0.3*aspect_score
        
        if total_score > best_score:
            best_score = total_score
            best_contour = cnt
    
    # Create final mask with edge refinement
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    
    # Edge-preserving refinement
    mask = cv2.ximgproc.guidedFilter(img, mask, radius=10, eps=100)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Final contour extraction
    final_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw result
    result = img.copy()
    cv2.drawContours(result, final_contours, -1, (0,255,0), 3)
    
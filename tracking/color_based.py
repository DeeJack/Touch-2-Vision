import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("videos/video.mp4")

"""
    Human Skin Detection Using RGB, HSV And Ycbcr Color Models https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
    
    0.0 <= H <= 50.0 and 0.23 <= S <= 0.68 and 
    R > 95 and G > 40 and B > 20 and R > G and R > B 
    and | R - G | > 15 and A > 15
"""
# lower_skin_rgb = np.array([95, 40, 20], dtype=np.uint8)
# lower_skin_hsv = np.array([0, 58, 30], dtype=np.uint8)
# lower_skin_ycbcr = np.array([80, 133, 77], dtype=np.uint8)

# upper_skin_rgb = np.array([255, 255, 255], dtype=np.uint8)
# upper_skin_hsv = np.array([127, 173, 230], dtype=np.uint8)
# upper_skin_ycbcr = np.array([255, 173, 127], dtype=np.uint8)

"""
    OR 
    R > 95 and G > 40 and B > 20 and R > G and R > B 
    and | R - G | > 15 and A > 15 and Cr > 135 and 
    Cb > 85 and Y > 80 and Cr <= (1.5862*Cb)+20 and 
    Cr>=(0.3448*Cb)+76.2069 and 
    Cr >= (-4.5652*Cb)+234.5652 and 
    Cr <= (-1.15*Cb)+301.75 and 
    Cr <= (-2.2857*Cb)+432.85nothing 
    (H : Hue ; S: Saturation ; R : Red ; B: Blue ; G : Green ; Cr, Cb : Chrominance components ; Y : luminance 
    component )
"""


# Define range of skin color in HSV
lower_skin_hsv = np.array([0, 10, 60], dtype=np.uint8)
upper_skin_hsv = np.array([20, 150, 255], dtype=np.uint8)

# Initialize previous frame
prev_frame = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
    # mask_rgb = cv2.inRange(rgb, lower_skin_rgb, upper_skin_rgb)
    # mask_ybcr = cv2.inRange(ycbcr, lower_skin_ycbcr, upper_skin_ycbcr)
    
    # Combine the masks
    # mask = cv2.bitwise_and(mask_hsv, mask_rgb)
    # mask = cv2.bitwise_and(mask, mask_ybcr)

    # Apply a series of morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for area
    min_area = 1000
    max_area = 10000
    contours = [c for c in contours if cv2.contourArea(c) > min_area and cv2.contourArea(c) < max_area]

    # If contours are found
    if contours:
        # Get the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)
        # Draw a rectangle around the hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

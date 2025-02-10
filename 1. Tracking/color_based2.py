"""
    Another attempt based on https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
    Didn't work.
"""

import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("videos/video.mp4")

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
# Initialize previous frame
prev_frame = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    skin_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # For each pixel, examine the colors
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            # Get the pixel values
            pixel_hsv = hsv[i, j]
            pixel_rgb = rgba[i, j]
            pixel_ycbcr = ycbcr[i, j]

            # Check if the pixel is skin
            """
                Human Skin Detection Using RGB, HSV And Ycbcr Color Models https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
                
                0.0 <= H <= 50.0 and 0.23 <= S <= 0.68 and 
                R > 95 and G > 40 and B > 20 and R > G and R > B 
                and | R - G | > 15 and A > 15
                
                Translated to 255:
                H <= (50 * 255) / 100 = 127.5
                S >= (0.23 * 255) = 58.65
                S <= (0.68 * 255) = 173.4
                
                R > 95, G > 40, B > 20
                
                |R - G| > 15
                
                A > 15
            """
            # if (
            #     pixel_rgb[0] > 95
            #     and pixel_rgb[1] > 40
            #     and pixel_rgb[2] > 20
            #     and pixel_rgb[0] > pixel_rgb[1]
            #     and pixel_rgb[0] > pixel_rgb[2]
            #     and abs(pixel_rgb[0] - pixel_rgb[1]) > 15
            #     and pixel_rgb[3] > 15
            #     and pixel_hsv[0] <= 127.5
            #     and pixel_hsv[0] >= 0
            #     and pixel_hsv[1] >= 58.65
            #     and pixel_hsv[1] <= 173.4
            # ):
            #     skin_mask[i, j] = 255
            """
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
            # if (
            #     pixel_rgb[0] > 95
            #     and pixel_rgb[1] > 40
            #     and pixel_rgb[2] > 20
            #     and pixel_rgb[0] > pixel_rgb[1]
            #     and pixel_rgb[0] > pixel_rgb[2]
            #     and abs(pixel_rgb[0] - pixel_rgb[1]) > 15
            #     and pixel_rgb[3] > 15
            #     and pixel_ycbcr[1] > 85
            #     and pixel_ycbcr[2] > 135
            #     and pixel_ycbcr[0] > 80
            #     and pixel_ycbcr[2] <= 1.5862 * pixel_ycbcr[1] + 20
            #     and pixel_ycbcr[2] >= 0.3448 * pixel_ycbcr[1] + 76.2069
            #     and pixel_ycbcr[2] >= -4.5652 * pixel_ycbcr[1] + 234.5652
            #     and pixel_ycbcr[2] <= -1.15 * pixel_ycbcr[1] + 301.75
            #     and pixel_ycbcr[2] <= -2.2857 * pixel_ycbcr[1] + 432.85
            # ):
            #     skin_mask[i, j] = 255

    # Apply a series of morphological operations to remove noise
    skin_mask = cv2.erode(skin_mask, None, iterations=2)
    skin_mask = cv2.dilate(skin_mask, None, iterations=2)

    cv2.imshow("Skin", skin_mask)
    cv2.imshow("Original", frame)
    # Threshold the HSV image to get only skin color
    # mask = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
    # mask_rgb = cv2.inRange(rgb, lower_skin_rgb, upper_skin_rgb)
    # mask_ybcr = cv2.inRange(ycbcr, lower_skin_ycbcr, upper_skin_ycbcr)

    # Combine the masks
    # mask = cv2.bitwise_and(mask_hsv, mask_rgb)
    # mask = cv2.bitwise_and(mask, mask_ybcr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

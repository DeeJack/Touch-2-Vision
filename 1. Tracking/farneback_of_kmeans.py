import numpy as np
import cv2 as cv

"""
    Farneback optical flow with k-means clustering
"""

cap = cv.VideoCapture(cv.samples.findFile("videos/video.mp4"))
ret, frame1 = cap.read()
previousFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
flow = None
while True:
    ret, frame2 = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    next_frame_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        previousFrame, next_frame_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Compute the magnitude and angle of the 2D vectors
    magnitudes, angles = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Calculate the mean magnitude and angle
    mean_mag = np.mean(magnitudes)
    mean_angle = np.mean(angles)
    
    height, width, _ = flow.shape
    flow_reshaped = flow.reshape((-1, 2))

    # Apply k-means to group pixels with similar motion.
    num_clusters = 4
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv.kmeans(
        flow_reshaped.astype(np.float32),
        num_clusters,
        None,
        criteria,
        10,
        cv.KMEANS_RANDOM_CENTERS,
    )
    
    # Reshape labels back to the original image shape to visualize the clustering result
    labels_image = labels.reshape((height, width))

    # Optionally, create a mask for a specific cluster (e.g., cluster 0)
    background = np.where(labels_image == 1, 255, 0).astype(np.uint8)
    foreground = np.where(labels_image == 0, 255, 0).astype(np.uint8)
    
    # Visualize the mask
    cv.imshow('Background', background)
    cv.imshow('Foreground', foreground)

    # Find regions where the magnitude of motion is significantly different from the mean
    mag_threshold = mean_mag * 2

    # Create a mask where the magnitude of motion is significantly different from the mean
    foreground_mask = magnitudes > (mean_mag + mag_threshold)
    
    foreground_mask = foreground_mask.astype(np.float32)

    # Convert the boolean mask to an 8-bit image
    foreground_mask = (foreground_mask * 255).astype(np.uint8)

    # Use morphological operations
    foreground_mask = cv.erode(foreground_mask, None, iterations=3)
    foreground_mask = cv.dilate(foreground_mask, None, iterations=3)


    
    # Optionally, visualize the optical flow
    # hsv = np.zeros_like(frame1)
    # hsv[..., 1] = 255
    # hsv[..., 0] = angle * 180 / np.pi / 2
    # hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # cv.imshow("frame2", res2)
    # cv.imshow("frame3", background)

    cv.imshow("frame", frame2)
    keyPressed = cv.waitKey(1) & 0xFF
    if keyPressed == ord("q"):
        break
    previousFrame = next_frame_gray

cv.destroyAllWindows()

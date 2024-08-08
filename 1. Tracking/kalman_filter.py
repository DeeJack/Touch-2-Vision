import cv2 as cv
import numpy as np

"""
    Kalman filter, don't know how to use it
"""

cap = cv.VideoCapture(cv.samples.findFile("videos/video.mp4"))
ret, frame1 = cap.read()
firstFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

while True:
    roi = cv.selectROI(frame1)
    x, y, w, h = roi

    ret, frame1 = cap.read()

    if roi is not None and (x != 0 and y != 0 and w != 0 and h != 0):
        break

# Set up the ROI for tracking
x, y, w, h = roi

track_window = (x, y, w, h)

# Set up the ROI for tracking
roi = frame1[y : y + h, x : x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

cv.imshow("aoi", roi)
cv.waitKey(10000)

roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

contours, _ = cv.findContours(roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv.contourArea)

# second_largest = sorted(contours, key=cv.contourArea)[-2]

# Create a mask for the largest contour
mask = np.zeros_like(frame1)
# cv.drawContours(mask, [largest_contour], -1, 255, -1)
cv.drawContours(mask, [largest_contour], -1, (255, 255, 255), 2)

cv.imshow("mask", mask)
cv.waitKey(10000)

# def kalman_filter_init():
#     # Initialize the Kalman filter
#     kalman = cv.KalmanFilter(4, 2)
#     kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
#     kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#     kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
#     kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
#     kalman.errorCovPost = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

#     return kalman

# kalman = kalman_filter_init()

# while cap.isOpened():
#     ret, frame2 = cap.read()
#     if not ret:
#         print("No frames grabbed!")
#         break

#     # Predict the new location
#     prediction = kalman.predict()

#     # Show the prediction
#     cv.circle(frame2, (int(prediction[0]), int(prediction[1])), 5, (0, 255, 0), -1)

#     # Update the measurement
#     # measurement = np.array([[np.float32(x)], [np.float32(y)]])
#     # kalman.correct(measurement)

#     cv.imshow('asd', frame2)

#     next_frame_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

import cv2 as cv
import numpy as np
import dotenv
import os
import matplotlib.pyplot as plt

dotenv.load_dotenv()

video = os.path.join(os.getenv("VIDEO_PATH"), os.getenv("VIDEO_NAME"))

mog_options = {"history": 100, "varThreshold": 16, "detectShadows": False}

mog_subtractor = cv.createBackgroundSubtractorMOG2(**mog_options)

video = cv.VideoCapture(video)

# Area threshold for the contours
min_area = 1600
max_area = 2600

# Define range of skin color in HSV
lower_skin = np.array([0, 10, 80], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    grey_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    ycbcr = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

    fg_mask = mog_subtractor.apply(ycbcr, learningRate=0.1)

    # Apply morphological operations to remove noise
    fg_mask = cv.erode(fg_mask, None, iterations=2)
    fg_mask = cv.dilate(fg_mask, None, iterations=2)

    # Find the contours of the foreground objects
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter them based on the area
    contours = [
        contour
        for contour in contours
        if cv.contourArea(contour) > min_area and cv.contourArea(contour) < max_area
    ]

    # if len(contours) > 0:
    #     print("Number of contours:", len(contours))

    for contour in contours:
        # Calculate the bounding box of the contour
        x, y, w, h = cv.boundingRect(contour)

        # Extract region of interest from the original frame
        roi = frame[y : y + h, x : x + w]
        
        # Convert frame to HSV color space
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        rgba = cv.cvtColor(roi, cv.COLOR_BGR2RGBA)
        ycbcr = cv.cvtColor(roi, cv.COLOR_BGR2YCrCb)

        skin_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # For each pixel, examine the colors
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                # Get the pixel values
                pixel_hsv = hsv[i, j]
                pixel_rgb = rgba[i, j]
                pixel_ycbcr = ycbcr[i, j]
                if (
                    pixel_rgb[0] > 95
                    and pixel_rgb[1] > 40
                    and pixel_rgb[2] > 20
                    and pixel_rgb[0] > pixel_rgb[1]
                    and pixel_rgb[0] > pixel_rgb[2]
                    and abs(pixel_rgb[0] - pixel_rgb[1]) > 15
                    and pixel_rgb[3] > 15
                    and pixel_hsv[0] <= 127.5
                    and pixel_hsv[0] >= 0
                    and pixel_hsv[1] >= 58.65
                    and pixel_hsv[1] <= 173.4
                ):
                    skin_mask[i, j] = 255
        
        if not skin_mask.any():
            continue
        print('Hand found!')
        cv.imshow('Hand', skin_mask)

        # mean_color = np.mean(roi, axis=(0, 1))
        # std_color = np.std(roi, axis=(0, 1))
        
        # print('Mean color:', mean_color, 'Std color:', std_color)

        # data = np.vstack((mean_color, std_color))

        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # k = 2  # Number of clusters
        # compactness, labels, centers = cv.kmeans(
        #     np.float32(data), k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        # )

        # print(labels)

        # hsv_row = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # mask = cv.inRange(hsv_row, lower_skin, upper_skin)

        # if mask.any():
        #     # Show a rect where the mask is present
        #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Calculate color features (e.g., histogram)
        # hist = cv.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # hist = cv.normalize(hist, hist).flatten()
        # plt.plot(hist, color='r')
        # plt.xlim([0,256])

        # plt.subplot(121), plt.hist(roi.ravel(), 256, [0, 256]), plt.title('ROI')

        # Perform object labeling based on color features

        # Perform blob extraction

        # Perform proximity-based tracking

        # Draw bounding box around the tracked object
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Original", frame)
    cv.imshow("Frame", fg_mask)
    # plt.show()

    if cv.waitKey(30) & 0xFF == ord("q"):  # ESC key
        break

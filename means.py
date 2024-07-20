import cv2
import numpy as np


def calculate_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None, **lk_params
    )

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        curr_frame = cv2.circle(curr_frame, (a, b), 5, (0, 255, 0), -1)

    return mask, curr_frame


def kmeans_segmentation(frame):
    # Convert image to feature space
    pixels = frame.reshape((-1, 3))

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Reshape labels to the size of the original image
    segmented_image = labels.reshape(frame.shape[0], frame.shape[1])

    return segmented_image


# Open video capture
cap = cv2.VideoCapture("videos/video.mp4")

# Read the first frame
ret, frame = cap.read()
prev_frame = frame.copy()

# Create a mask and initialize previous points
mask = np.zeros_like(frame)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(
    prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate optical flow
    mask, frame_with_flow = calculate_optical_flow(prev_frame, frame)

    # Apply K-means segmentation
    segmented_image = kmeans_segmentation(frame)

    # Display results
    cv2.imshow("Optical Flow", frame_with_flow)
    cv2.imshow("K-Means Segmentation", (segmented_image * 255).astype(np.uint8))

    # Exit on ESC
    if cv2.waitKey(30) == 27:
        break

    # Update previous frame and points
    prev_frame = frame.copy()
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        mask=None,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

cap.release()
cv2.destroyAllWindows()

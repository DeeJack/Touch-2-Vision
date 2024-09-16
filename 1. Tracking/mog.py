"""
    Only MoG
"""

import cv2

video = cv2.VideoCapture("videos/video.mp4")
outputWriter = cv2.VideoWriter('./results/mog.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(video.get(3)), int(video.get(4))), True)
N_GAUSS = 5
BACKGROUND_THRESHOLD = 0.8
NOISE_SIGMA = 1
HISTORY = 100  # t
ALPHA = 0.5
mog_subtractor = cv2.createBackgroundSubtractorMOG2()
min_area_threshold = 1100
max_area_threshold = 3000
# mog_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(HISTORY, N_GAUSS, BACKGROUND_THRESHOLD, NOISE_SIGMA)
# knn_subtractor = cv2.createBackgroundSubtractorKNN(history= HISTORY)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # lab
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # threshold
    thresh = cv2.inRange(l, 40, 120)

    # contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by size
    marked = frame.copy()
    cv2.drawContours(marked, contours, -1, (0, 255, 0), 3)

    # show
    cv2.imshow("marked", marked)
    cv2.imshow("Thresh", thresh)
    outputWriter.write(marked)
    cv2.waitKey(1)
    continue
video.release()
outputWriter.release()
cv2.destroyAllWindows()
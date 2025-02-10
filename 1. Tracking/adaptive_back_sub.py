"""
    Trying Adaptive Background Subtraction
"""

import numpy as np
import cv2

video = cv2.VideoCapture("./videos/video.mp4")
outputWriter = cv2.VideoWriter(
    "./results/adaptive_back_sub.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (int(video.get(3)), int(video.get(4))),
    True,
)
frames = []
N = 10
t = 0

THRESH = 50
MAXVAL = 255
ALPHA = 0.8
background = None

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames.append(frame_greyscale)

    if background is None:
        background = frame_greyscale

    if t > N:
        diff = cv2.absdiff(frame_greyscale, background)
        _, frame_diff_thresh = cv2.threshold(diff, THRESH, MAXVAL, cv2.THRESH_BINARY)

        # Adjust the background with BG = alpha * frame + (1 - alpha) * BG
        # Alpha dictates how much the background is able to adapt to changes.
        background = np.uint8(ALPHA * frame_greyscale + (1 - ALPHA) * background)

        cv2.imshow("dsa", frame_greyscale)
        cv2.imshow("background", background)
        cv2.imshow("foreground", frame_diff_thresh)
        frame_rgb = cv2.cvtColor(frame_diff_thresh, cv2.COLOR_GRAY2RGB)
        outputWriter.write(frame_rgb)
        cv2.waitKey(1)
    t += 1
outputWriter.release()

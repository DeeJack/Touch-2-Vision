"""
    Inpaint using OpenCV's inpainting function. Produces poor results.
"""

import os
import cv2

file_name = "video.mp4"
videos_path = "videos"
mask_path = "results"
mask_name = "mediapipe_mask.mp4"


video = cv2.VideoCapture(os.path.join(videos_path, file_name))
mask = cv2.VideoCapture(os.path.join(mask_path, mask_name))
outputWriter = cv2.VideoWriter(
    "results/inpaint_cv.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (int(video.get(3)), int(video.get(4))),
    True,
)

while True:
    ret, frame = video.read()
    ret, mask_frame = mask.read()

    if not ret:
        break

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask_frame)

    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    # result = cv2.inpaint(frame, mask_frame, 3, cv2.INPAINT_NS)
    result = cv2.inpaint(frame, mask_frame, 3, cv2.xphoto.INPAINT_FSR_BEST)

    cv2.imshow("result", result)
    outputWriter.write(result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
mask.release()
outputWriter.release()
cv2.destroyAllWindows()

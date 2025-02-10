"""
Extract frames from a video file and save them as JPEG images.

The script takes a video file as input and extracts each frame, optionally resizing them
based on SCALE_PERCENTAGE. The frames are saved as JPEG images with quality=85 in the
specified TARGET_FOLDER.

Parameters:
    VIDEO_TO_EXTRACT: cv2.VideoCapture object
        The input video to extract frames from
    TARGET_FOLDER: str 
        Directory path where extracted frames will be saved
    SCALE_PERCENTAGE: int
        Percentage to scale frames (100 = original size)
"""

import cv2 as cv
import os

VIDEO_TO_EXTRACT: cv.VideoCapture = cv.VideoCapture("./videos/inpaint_out.mp4")
TARGET_FOLDER: str = "./inpainted_frames"
SCALE_PERCENTAGE: int = 100  # percentage of original size
JPEG_QUALITY: int = 85

count: int = 0

if not os.path.isdir(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)

while VIDEO_TO_EXTRACT.isOpened():
    ret, frame = VIDEO_TO_EXTRACT.read()

    if not ret:
        print("End of video")
        break

    # Resize frame
    width = int(frame.shape[1] * SCALE_PERCENTAGE / 100)
    height = int(frame.shape[0] * SCALE_PERCENTAGE / 100)
    dim = (width, height)
    resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    # Save frame as JPEG with quality 85
    cv.imwrite(
        os.path.join(TARGET_FOLDER, f"frame_{count:0000}.jpg"),
        resized_frame,
        [int(cv.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    count += 1

VIDEO_TO_EXTRACT.release()

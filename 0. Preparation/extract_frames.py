import cv2 as cv
import os

original_video = cv.VideoCapture('./videos/20220607_133934/gelsight.mp4')
count = 0
scale_percent = 50  # percentage of original size

TARGET_FOLDER = './gelsight_frames'

if not os.path.isdir(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)

while original_video.isOpened():
    ret, frame = original_video.read()
    
    if not ret:
        print("End of video")
        break
    
    # Resize frame
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    
    # Save frame as JPEG with quality 85 (scale of 0-100, higher means better quality)
    cv.imwrite(os.path.join(TARGET_FOLDER, f'frame_{count:0000}.jpg'), resized_frame, [int(cv.IMWRITE_JPEG_QUALITY), 75])
    count += 1
    cv.waitKey(1)

original_video.release()
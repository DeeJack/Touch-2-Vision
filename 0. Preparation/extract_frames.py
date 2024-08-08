import cv2 as cv

original_video = cv.VideoCapture('videos/video.mp4')
count = 0
scale_percent = 50  # percentage of original size

while original_video.isOpened():
    ret, frame = original_video.read()
    
    if not ret:
        break
    
    # Resize frame
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    
    # Save frame as JPEG with quality 85 (scale of 0-100, higher means better quality)
    cv.imwrite(f'frames/frame_{count:0000}.jpg', resized_frame, [int(cv.IMWRITE_JPEG_QUALITY), 85])
    count += 1
    cv.waitKey(1)

original_video.release()
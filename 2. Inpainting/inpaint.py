import os
import cv2

file_name = 'video.mp4'
videos_path = 'videos'
mask_path = 'results'
mask_name = 'hand_' + file_name


video = cv2.VideoCapture(os.path.join(videos_path, file_name))
mask = cv2.VideoCapture(os.path.join(mask_path, mask_name))

while True:
    ret, frame = video.read()
    ret, mask_frame = mask.read()
    
    if not ret:
        break
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask_frame)
    
    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    # result = cv2.inpaint(frame, mask_frame, 3, cv2.INPAINT_NS)
    result = cv2.inpaint(frame, mask_frame, 3, cv2.xphoto.INPAINT_FSR_BEST)
    
    cv2.imshow('result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
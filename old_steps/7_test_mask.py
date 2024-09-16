import cv2 as cv

mask_folder = "masks"
video_path = "videos/video.mp4"

cap = cv.VideoCapture(video_path)

frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    mask = cv.imread(f"{mask_folder}/mask_{frame_num:04d}.png", cv.IMREAD_GRAYSCALE)
    
    # Color the area covered by the mask in green
    frame[mask > 0] = [0, 255, 0]

    cv.imshow("Frame", frame)
    
    frame_num += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
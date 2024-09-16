"""
    Trying Farneback optical flow
"""

import numpy as np
import cv2 as cv

video = cv.VideoCapture(cv.samples.findFile("videos/video.mp4"))
outputWriter = cv.VideoWriter('./results/farneback_optical_flow.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (int(video.get(3)), int(video.get(4))), True)
ret, frame1 = video.read()
previousFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
flow = None
while True:
    ret, frame2 = video.read()
    if not ret:
        print("No frames grabbed!")
        break

    next_frame_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(previousFrame, next_frame_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angleRadians = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angleRadians * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    foregroundMaskedImage = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # foregroundMaskedImage = cv.cvtColor(foregroundMaskedImage, cv.COLOR_BGR2GRAY)
    
    overlay = cv.addWeighted(frame2, 0.5, foregroundMaskedImage, 0.5, 0)
    
    # contours, _ = cv.findContours(
    #     foregroundMaskedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    # )
    
    # for contour in contours:
    #     x, y, w, h = cv.boundingRect(contour)
    #     cv.drawContours(next_frame, contour, -1, (0, 255, 0), 3)
    
    cv.imshow("frame", overlay)
    outputWriter.write(overlay)
    
    # cv.imshow("frame2", hsv)
    keyPressed = cv.waitKey(1) & 0xFF
    if keyPressed == ord("q"):
        break
    elif keyPressed == ord("s"):
        cv.imwrite("opticalfb.png", frame2)
        cv.imwrite("opticalhsv.png", foregroundMaskedImage)
    previousFrame = next_frame_gray

cv.destroyAllWindows()
outputWriter.release()
video.release()

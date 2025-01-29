import torch
import cv2

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True) # or any other size:  'yolov5m', 'yolov5l', 'yolov5x'

# Load your image
img = cv2.imread('gelsight_frames/frame_0.jpg')

# Perform inference
results = model(img)

# Print results
results.print()

print(results.pandas().xyxy[0])

# Visualize results (optional)
# results.show()
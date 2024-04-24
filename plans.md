# Touch to vision

1. Background Subtraction
2. Look into the features of the objects inside the regions obtained (hist)

Problem: the objects can be made up of several blobs. Join blobs with similar features and near each other

Do not track on frame-basis, use the closest blob at each frame

(background suppression?)

Optical Flow: Optical flow methods track the movement of pixels between consecutive frames. These methods can handle camera motion to some extent but may struggle with complex scenes and occlusions.

Feature-Based Methods: Feature-based methods detect and track keypoints or feature points in the scene. Techniques like SIFT, SURF, or ORB can be used for feature detection and matching across frames.

Deep Learning-based Approaches: Deep learning models, such as convolutional neural networks (CNNs), can be trained to detect and track objects in video sequences. Models like YOLO (You Only Look Once) or Mask R-CNN are commonly used for object detection and tracking.

Kalman Filtering: Kalman filters can be used to estimate the state of moving objects in a video sequence, including both the object's position and velocity. Kalman filters can help in predicting the object's position even when the camera is moving.

Foreground-Background Segmentation: Instead of relying on a fixed background model, dynamic background subtraction methods can be used to model the scene's background adaptively. These methods can handle gradual changes in the background due to camera motion or lighting variations. [URL](https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/)

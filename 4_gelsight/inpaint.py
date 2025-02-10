"""
    Tries to inpaint the black artifacts in the Gelsight images.
"""

import cv2
import numpy as np


def remove_gelsight_artifacts(image):
    """
    Removes circular black artifacts from a Gelsight image using blob detection and inpainting.

    Args:
        image (numpy.ndarray): The input Gelsight image (BGR).

    Returns:
        numpy.ndarray: The inpainted image with artifacts removed.
    """

    # 1. Convert to grayscale for blob detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.4
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    detector = cv2.SimpleBlobDetector_create(params)

    # 4. Detect blobs (the artifacts)
    keypoints = detector.detect(gray)

    # 5. Create a mask of the artifacts
    mask = np.zeros_like(gray, dtype=np.uint8)
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        r = int(keypoint.size / 2)  # Radius from blob size
        cv2.circle(mask, (x, y), r, 255, -1)  # Draw filled circle in mask

    # 6. Inpaint the artifacts (if any)
    if np.any(mask):
        blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)
        inpainted_image = cv2.inpaint(
            image, blurred_mask, inpaintRadius=3, flags=cv2.INPAINT_NS
        )
    else:
        inpainted_image = image.copy()

    return inpainted_image


def process_video(video_path, output_path=None, display=True):
    """
    Processes a video, removing artifacts from each frame.

    Args:
        video_path (str): The path to the input video file.
        output_path (str, optional): The path to save the processed video. If None, the video isn't saved.
        display (bool, optional): Whether to display the processed video.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        cleaned_frame = remove_gelsight_artifacts(frame)

        if output_path:
            out.write(cleaned_frame)

        if display:
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Cleaned Frame", cleaned_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Press 'q' to exit

    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Video processed, number of frames: {frame_count}")


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    output_video_path = "results/gelsight_result.mp4"
    process_video(video_path, output_video_path, display=True)

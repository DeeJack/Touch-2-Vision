import cv2
import numpy as np


class ArtifactTracker:
    def __init__(self, alpha_stable=0.9, alpha_moving=0.3, movement_threshold=5):
        self.tracked_artifacts = []  # List of (x, y, radius)
        self.alpha_stable = alpha_stable  # Smoothing factor for stable artifacts
        self.alpha_moving = alpha_moving  # Smoothing factor for moving artifacts
        self.movement_threshold = movement_threshold  # Threshold for detecting movement

    def update(self, detected_keypoints):
        """
        Updates the tracked artifact positions based on new detections.
        """
        new_artifacts = []

        # Match new detections to existing tracked artifacts
        for keypoint in detected_keypoints:
            x, y = keypoint.pt
            r = keypoint.size / 2

            # Find closest tracked artifact (if any)
            closest_artifact = None
            min_distance = float("inf")
            for i, (tx, ty, tr) in enumerate(self.tracked_artifacts):
                distance = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_artifact = i

            # Update or add new artifact
            if (
                closest_artifact is not None and min_distance < 2 * r
            ):  # Threshold distance
                tx, ty, tr = self.tracked_artifacts[closest_artifact]
                # Adaptively adjust alpha based on movement
                if min_distance > self.movement_threshold:
                    alpha = self.alpha_moving
                else:
                    alpha = self.alpha_stable
                new_x = alpha * tx + (1 - alpha) * x
                new_y = alpha * ty + (1 - alpha) * y
                new_r = alpha * tr + (1 - alpha) * r

                new_artifacts.append((new_x, new_y, new_r))
                self.tracked_artifacts.pop(
                    closest_artifact
                )  # remove from the previous artifacts list
            else:
                # Add as a new artifact if no close match found
                new_artifacts.append((x, y, r))
        # add the previous artifacts in the new_artifacts list
        new_artifacts.extend(self.tracked_artifacts)
        self.tracked_artifacts = new_artifacts

    def get_mask(self, frame_shape):
        """
        Generates a mask based on the current tracked artifact positions.
        """
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for x, y, r in self.tracked_artifacts:
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        return mask


def remove_gelsight_artifacts(image, artifact_tracker):
    """
    Removes circular black artifacts from a Gelsight image using blob detection and inpainting, using tracked artifact positions.
    """

    # Convert to grayscale for blob detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pre-processing steps
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)  # use enhanced_gray for blob detection

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    # Filter by color (black)
    params.filterByColor = True
    # params.blobColor = 0  # 0 for black blobs

    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.4
    params.filterByInertia = False
    # params.minInertiaRatio = 0.2

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs (the artifacts)
    keypoints: tuple = detector.detect(enhanced_gray)
    keypoints = list(keypoints)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,  # adjust minDist
        param1=50,
        param2=20,
        minRadius=2,
        maxRadius=50,
    )  # adjust params and radii
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        hough_keypoints = []
        for x, y, r in circles:
            hough_keypoints.append(
                cv2.KeyPoint(float(x), float(y), 2 * float(r))
            )  # Create KeyPoint object for consistency
        keypoints.extend(hough_keypoints)  # Combine blob and hough keypoints

    # Update tracked artifact positions
    artifact_tracker.update(keypoints)

    # Get mask from tracker
    mask = artifact_tracker.get_mask(gray.shape)

    # Pre-process the mask
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    # Apply blurring to the mask
    blurred_mask = cv2.GaussianBlur(closed_mask, (5, 5), 0)

    kernel = np.ones((3, 3), np.uint8)  # adjust kernel size
    opened_gray = cv2.morphologyEx(
        blurred_mask, cv2.MORPH_OPEN, kernel
    )  # use opened_gray for blob detection

    # Inpaint the artifacts (if any)
    if np.any(mask):
        inpainted_image = cv2.inpaint(
            image, opened_gray, inpaintRadius=3, flags=cv2.INPAINT_NS
        )
    else:
        inpainted_image = image.copy()

    return inpainted_image


def process_video(video_path, output_path=None, display=False):
    """
    Removes artifacts from each frame, using artifact tracking.
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

    artifact_tracker = ArtifactTracker(
        alpha_stable=0.9, alpha_moving=0.3, movement_threshold=3
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        cleaned_frame = remove_gelsight_artifacts(frame, artifact_tracker)

        height, width = cleaned_frame.shape[:2]
        crop_percentage = 0.01
        x1 = int(width * crop_percentage)
        y1 = int(height * crop_percentage)
        x2 = int(width * (1 - crop_percentage))
        y2 = int(height * (1 - crop_percentage))
        cropped_frame = cleaned_frame[y1:y2, x1:x2]

        if output_path:
            out.write(cleaned_frame)

        if display:
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Cropped Image", cropped_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Video processed, number of frames: {frame_count}")


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    output_video_path = "results/inpaint2.mp4"
    process_video(video_path, output_video_path, display=False)

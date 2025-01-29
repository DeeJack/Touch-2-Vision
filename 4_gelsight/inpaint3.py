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
            min_distance = float('inf')
            for i, (tx, ty, tr) in enumerate(self.tracked_artifacts):
                distance = ((x - tx)**2 + (y - ty)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_artifact = i

            # Update or add new artifact
            if closest_artifact is not None and min_distance < 2 * r:  # Threshold distance
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
                self.tracked_artifacts.pop(closest_artifact) # remove from the previous artifacts list
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

def remove_gelsight_artifacts(image, artifact_tracker, lower_color, upper_color):
    """
    Removes circular artifacts from a Gelsight image using blob detection and inpainting,
    filtering by a specified color range.
    """

    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Create a mask for the specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 3. Find Contours in the Mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Convert Contours to Keypoints
    keypoints = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 0:  # Ensure the radius is valid
            keypoint = cv2.KeyPoint(x, y, 2 * radius)  # Size is diameter
            keypoints.append(keypoint)

    # 5. Update tracked artifact positions
    artifact_tracker.update(keypoints)

    # 6. Get mask from tracker
    tracked_mask = artifact_tracker.get_mask(mask.shape)

    # 7. Pre-process the mask
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    dilated_mask = cv2.dilate(tracked_mask, kernel, iterations=1)
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    # Apply blurring to the mask
    blurred_mask = cv2.GaussianBlur(closed_mask, (5, 5), 0)  # Adjust kernel size as needed

    # 8. Inpaint the artifacts (if any)
    if np.any(tracked_mask):
        inpainted_image = cv2.inpaint(image, blurred_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    else:
        inpainted_image = image.copy()

    return inpainted_image


def process_video(video_path, lower_color, upper_color, output_path=None, display=True):
    """
    Processes a video, removing artifacts from each frame, using artifact tracking and color range filtering.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use another codec
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize artifact tracker
    artifact_tracker = ArtifactTracker(alpha_stable=0.9, alpha_moving=0.3, movement_threshold=5)  # Adjust parameters as needed

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        cleaned_frame = remove_gelsight_artifacts(frame, artifact_tracker, lower_color, upper_color)

        if output_path:
            out.write(cleaned_frame)

        if display:
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Cleaned Frame', cleaned_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit

    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()
    print(f"Video processed, number of frames: {frame_count}")
# Example usage

if __name__ == '__main__':
    # Example usage
    video_path = 'videos/20220607_133934/gelsight.mp4'  # Replace with your video file
    output_video_path = 'results/inpaint3_output.mp4'  # If None, the video is not saved.

    # Define the color range to filter (HSV format)
    lower_color = np.array([0, 0, 0])  # Example: Lower bound for dark gray/black (H, S, V)
    upper_color = np.array([0, 150, 150])  # Example: Upper bound for dark gray/black (H, S, V)
    process_video(video_path, lower_color, upper_color, output_video_path, display=True)
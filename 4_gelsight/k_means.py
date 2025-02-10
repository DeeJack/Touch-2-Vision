"""
Apply k-means clustering to segment a GelSight sensor image into k distinct regions.
"""

import cv2
import numpy as np


def advanced_segmentation(image_path, k=3):
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels and convert to float32.
    pixel_vals = image_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # Define stopping criteria and apply k-means.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers back to 8-bit values.
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image_rgb.shape))

    # Display the overall segmented image.
    cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    # Display each cluster as a separate binary mask.
    labels_reshaped = labels.flatten().reshape((image.shape[0], image.shape[1]))
    for i in range(k):
        # Create a binary mask for cluster i.
        cluster_mask = np.uint8((labels_reshaped == i) * 255)
        cv2.imshow(f"Cluster {i}", cluster_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fps = 27
    frame_num = (8 * 60) * fps + 40  # 8 min, 7 seconds
    target_frame = f"./gelsight_frames/frame_{frame_num}.jpg"
    advanced_segmentation(target_frame, k=3)

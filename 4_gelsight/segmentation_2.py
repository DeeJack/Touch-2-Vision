"""
    Uses segmentation again, taking the largest area of the segmented image.
"""

import cv2
import numpy as np


def advanced_segmentation_get_largest_area(image_path, k=3):
    # Load the image.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Convert the image from BGR to RGB (for consistency with k-means segmentation).
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels and convert to float32.
    pixel_vals = image_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # Define stopping criteria and apply k-means clustering.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    attempts = 10
    _, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers back to 8-bit values and reconstruct the segmented image.
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image_rgb.shape)

    # Find the cluster with the largest area (i.e. the most pixels).
    labels_flat = labels.flatten()
    counts = np.bincount(labels_flat)
    largest_cluster = np.argmax(counts)
    print(
        f"Largest cluster index: {largest_cluster} with {counts[largest_cluster]} pixels"
    )

    # Create a binary mask for the largest cluster.
    labels_reshaped = labels_flat.reshape((image.shape[0], image.shape[1]))
    mask = np.uint8((labels_reshaped == largest_cluster) * 255)

    # (Optional) Clean up the mask with morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    inverted_mask = cv2.bitwise_not(mask_clean)

    # Extract the region from the original image using the mask.
    extracted_region = cv2.bitwise_and(image, image, mask=inverted_mask)

    # Take the largest contour from the mask
    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    mask_clean = cv2.drawContours(
        mask_clean, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED
    )
    extracted_region = cv2.drawContours(
        extracted_region, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED
    )

    # Display the results.
    cv2.imshow(
        "Segmented Image (k-means)", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    )
    cv2.imshow("Largest Cluster Mask", mask_clean)
    cv2.imshow("Extracted Region from Original Image", extracted_region)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fps = 27
    frame_num = (8 * 60) * fps + 40  # 8 min, 7 seconds
    target_frame = f"./gelsight_frames/frame_{frame_num}.jpg"
    advanced_segmentation_get_largest_area(target_frame, k=3)

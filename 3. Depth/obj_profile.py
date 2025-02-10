"""
    Get the object profile by using the depth map, the estimated area
    covered by the sensor, and the object mask.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def visualize_object_altitude(
    image_path, depth_map_path, mask_path, altitude_threshold=0.1
):
    """
    Visualizes the object in the original image, highlighting regions that are higher in altitude
    relative to their surroundings based on depth information.

    Args:
        image_path: Path to the original image (e.g., a PNG or JPG file).
        depth_map_path: Path to the depth map image (e.g., a PNG or JPG file).
        mask_path: Path to the mask image (e.g., a PNG or JPG file).  The mask should have values of 0
                    for the background and 255 (or 1) for the object.
        altitude_threshold: The minimum depth difference to consider a region as "higher in altitude."
    """
    try:
        # Load the original image, depth map, and mask
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not load original image from {image_path}")
            return

        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            print(f"Error: Could not load depth map from {depth_map_path}")
            return

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not load mask from {mask_path}")
            return
        # Scale the mask down by 50%
        mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)

        # Check if the image has values from 0 to 1
        if np.max(depth_map) <= 1.0:
            print("The depth map has values between 0 and 1. Converting to 8-bit")
            depth_map = (depth_map * 255).astype(np.uint8)

        # Check if the image has values from 0 to 255
        if np.max(mask) <= 1.0:
            print("The mask has values between 0 and 1. Converting to 8-bit")
            mask = (mask * 255).astype(np.uint8)

        # Convert the mask to binary (0 or 1) if it's grayscale
        if len(mask.shape) == 2:
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)  # Binarize the mask
        else:
            print("Mask is not grayscale.  Please provide a grayscale mask image.")
            return

        # Ensure depth_map is of float type for calculations
        if depth_map.dtype != np.float32 and depth_map.dtype != np.float64:
            depth_map = depth_map.astype(np.float32)

        # Verify that the depth map and mask have the same dimensions
        if (
            depth_map.shape[:2] != mask.shape[:2]
            or original_image.shape[:2] != mask.shape[:2]
        ):
            print(
                f"Error: Depth map, mask, and original image must have the same dimensions. Dimensions: {depth_map.shape}, {mask.shape}, {original_image.shape}"
            )
            return

    except Exception as e:
        print(f"An error occurred during image loading or processing: {e}")
        return

    # Apply the mask to the depth map
    masked_depth_map = depth_map * mask

    # Calculate a local depth average for comparison.  Using a small kernel size.
    kernel_size = 5
    local_depth_average = cv2.blur(masked_depth_map, (kernel_size, kernel_size))

    # Highlight regions with depth significantly higher than the local average
    altitude_map = masked_depth_map - local_depth_average

    # Create a highlight mask based on the threshold
    highlight_mask = np.where(altitude_map > altitude_threshold, 1, 0).astype(
        np.uint8
    )  # Convert the boolean array into a 0 and 1 array.

    # Visualize the altitude by overlaying a color on the original image.
    color = [0, 0, 255]  # Blue color for highlighting (BGR format)
    intensity = 0.5  # Adjust the intensity of the highlighting

    highlighted_image = original_image.copy()

    # Apply the highlighting to the original image
    for c in range(3):  # Iterate through the BGR channels
        highlighted_image[:, :, c] = np.where(
            highlight_mask == 1,
            highlighted_image[:, :, c] * (1 - intensity) + color[c] * intensity,
            highlighted_image[:, :, c],
        )

    # Display the original image, highlighted image, mask and depth image
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    plt.title("Highlighted Image (Altitude Visualization)")

    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")

    plt.subplot(2, 2, 4)
    plt.imshow(depth_map, cmap="viridis")
    plt.colorbar()
    plt.title("Depth Map")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    frame = 70
    image_path = f"./inpainted_frames/frame_{frame}.jpg"
    depth_map_path = f"./depth_maps/depth_map_00{frame}.png"
    image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    if image is None:
        print("Error: Could not load the depth map.")
        exit

    mask_path = f"./masked_images/masked_frame_00{frame}.png"
    altitude_threshold = 0.1

    visualize_object_altitude(image_path, depth_map_path, mask_path, altitude_threshold)

"""
Try using various image processing techniques to extract object profiles from Gelsight sensor images.
This includes:
- Color space transformations (HSV, LAB) to better separate object features
- CLAHE contrast enhancement
- Channel visualization for analysis
- Contour detection and filtering
- Morphological operations for noise reduction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_gelsight_profile(image_path):
    """
    Extracts the object profile from a Gelsight sensor image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        tuple: (original_image, processed_image, object_mask, contours)
                - original_image: The original loaded image.
                - processed_image: The image after processing steps.
                - object_mask: Binary mask highlighting the object.
                - contours: List of contours detected.
    """

    # 1. Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None, None, None, None

    # 2. Convert to a color space that might be more helpful
    #    HSV or Lab often separate color and intensity information better than RGB
    # Convert to HSV and LAB color spaces
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 0])  # Enhance Value channel
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])  # Enhance Lightness channel

    # ---  Color Channel Exploration ---
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1), plt.imshow(
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    ), plt.title("Original RGB")
    plt.subplot(2, 3, 2), plt.imshow(hsv_image[:, :, 0], cmap="gray"), plt.title(
        "HSV - Hue"
    )
    plt.subplot(2, 3, 3), plt.imshow(hsv_image[:, :, 1], cmap="gray"), plt.title(
        "HSV - Saturation"
    )
    plt.subplot(2, 3, 4), plt.imshow(hsv_image[:, :, 2], cmap="gray"), plt.title(
        "HSV - Value"
    )
    plt.subplot(2, 3, 5), plt.imshow(lab_image[:, :, 0], cmap="gray"), plt.title(
        "Lab - L (Lightness)"
    )
    plt.subplot(2, 3, 6), plt.imshow(lab_image[:, :, 1], cmap="gray"), plt.title(
        "Lab - a (Green-Red)"
    )
    plt.show()

    # 3. Select the best channel for segmentation (Customize this!)
    #    processed_image = hsv_image[:,:,1]  # Saturation channel
    #    processed_image = lab_image[:,:,1]  # 'a' channel from Lab
    processed_image = hsv_image[:, :, 0]

    # 4. Noise Reduction (Blurring)
    # blurred_image = cv2.GaussianBlur(
    #     processed_image, (5, 5), 0
    # )  # Adjust kernel size (5,5) as needed
    blurred_image = processed_image

    # 5. Thresholding to create a binary mask
    #    Experiment with different thresholding methods and parameters.
    #    - cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, etc.
    #    - Adjust threshold values (e.g., 100, 150, etc.)

    # Example 1: Simple Thresholding
    _, thresholded_image = cv2.threshold(blurred_image, 130, 255, cv2.THRESH_BINARY)

    # Example 2: Adaptive Thresholding
    # thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Example 3: Otsu's Thresholding
    # _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Morphological Operations
    #    - Erosion to remove small noise
    #    - Dilation to connect broken parts of the object
    #    - Opening (Erosion followed by Dilation) to remove small objects
    #    - Closing (Dilation followed by Erosion) to close small holes

    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(
        thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2
    )  # Opening to remove noise
    # morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel, iterations=2) # Closing to fill gaps if needed

    # 7. Contour Detection
    contours, hierarchy = cv2.findContours(
        morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # RETR_EXTERNAL:  Get only the outer contours (useful for object profile).
    # CHAIN_APPROX_SIMPLE: Approximates contours to save memory.

    # 8. Contour Filtering (Optional but important for artifact removal)

    filtered_contours = []
    min_contour_area = 1000
    max_contour_area = 100000
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:  # Area based filtering
            filtered_contours.append(contour)

    # 9. Draw Contours on the original image (for visualization)
    contour_image = original_image.copy()
    cv2.drawContours(
        contour_image, filtered_contours, -1, (0, 255, 0), 2
    )  # Green contours

    return original_image, contour_image, morph_image, filtered_contours


if __name__ == "__main__":
    fps = 27
    frame_num = (8 * 60) * fps + 40  # 8 min, 7 seconds
    target_frame = f"./results/inpaint2.jpg"
    print(target_frame)
    original_img, contoured_img, mask_img, contours = extract_gelsight_profile(
        target_frame
    )

    if original_img is not None:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_img, cmap="gray")  # Display the mask
        plt.title("Object Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(contoured_img, cv2.COLOR_BGR2RGB))
        plt.title("Contours on Original Image")

        plt.tight_layout()
        plt.show()

        print(f"Number of contours detected after filtering: {len(contours)}")

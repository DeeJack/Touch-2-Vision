"""
    This one tries to remove the circular artifacts before the contour detection
"""

import cv2
import numpy as np


def remove_periodic_noise(img):
    """Remove circular artifacts using frequency domain filtering"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # FFT and shift
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create high-pass filter mask
    mask = np.ones((rows, cols), np.uint8)
    r = 60  # Radius to keep (adjust based on artifact size)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # Apply mask and inverse FFT
    fshift *= mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def extract_object_profile_v3(image_path):
    # Load image and remove periodic noise
    img = cv2.imread(image_path)
    denoised = remove_periodic_noise(img)

    # Enhanced color processing in LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on A and B channels
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    a = clahe.apply(a)
    b = clahe.apply(b)
    enhanced_lab = cv2.merge([l, a, b])

    # Adaptive color thresholding
    # Convert LAB to BGR
    bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Calculate color intensity (sum of BGR channels)
    intensity = np.sum(bgr, axis=2)

    # Create mask of highest intensity region
    threshold = np.percentile(intensity, 90)  # Take top 5% brightest pixels
    mask = intensity > threshold

    # Apply mask to get only highest intensity region
    # Convert mask to uint8 for morphological operations
    mask = mask.astype(np.uint8) * 255

    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply morphological operations to clean up the mask
    # Closing fills small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    # Opening removes small noise
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Dilate slightly to ensure coverage
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Convert back to boolean mask
    mask = mask > 0

    # Apply mask to BGR image
    bgr = bgr.copy()
    bgr[~mask] = 0
    # Create a copy of the original image
    original_img = img.copy()
    # Set everything outside the mask to black in the original image
    original_img[~mask] = 0

    return bgr


fps = 27
frames_path = "./gelsight_frames/"
target_frame = 7 * 60 + 8  # 08:03, 07:08
number = target_frame * fps
input_path = f"{frames_path}frame_{number}.jpg"
# "gelsight_frames/frame_70.jpg"

result_img = extract_object_profile_v3(input_path)

cv2.imshow("Result", result_img)
cv2.waitKey(10000)
# cv2.imshow("Mask", profile_mask)

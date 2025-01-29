import cv2
import numpy as np

def find_artifact_regions(image, artifact_templates):
    """
    Finds regions in an image that are similar to artifact templates.
    """

    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for the input image
    kp1, des1 = sift.detectAndCompute(image, None)
    if des1 is None:
        return []

    artifact_regions = []
    for template in artifact_templates:
        # Detect and compute keypoints and descriptors for the artifact template
        kp2, des2 = sift.detectAndCompute(template, None)
        if des2 is None:
            continue

        # Use a brute-force matcher to find matching keypoints
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m in matches:
            print(m)
            if len(m) < 2:
                continue
            if m[0].distance < 0.75 * m[1].distance:
                good_matches.append(m[0])

        # If enough good matches are found, consider it a potential artifact region
        if len(good_matches) > 10:  # Adjust threshold as needed
            # Estimate the homography matrix
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Draw a rectangle around the matched region
            h, w = template.shape
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if H is not None:
              try:
                dst = cv2.perspectiveTransform(pts, H)
                artifact_regions.append(np.int32(dst))
              except cv2.error as e:
                print(f"Homography transformation error: {e}")
                continue


    return artifact_regions

# Example Usage
if __name__ == '__main__':
    # Load your Gelsight image
    image = cv2.imread('gelsight_frames/frame_0.jpg', cv2.IMREAD_GRAYSCALE)

    # Load your artifact templates (grayscale)
    template1 = cv2.imread('C:\\Users\\loren\\Downloads\\artifact_template0.png', cv2.IMREAD_GRAYSCALE)
    template2 = cv2.imread('C:\\Users\\loren\\Downloads\\artifact_template1.png', cv2.IMREAD_GRAYSCALE)
    template3 = cv2.imread('C:\\Users\\loren\\Downloads\\artifact_template2.png', cv2.IMREAD_GRAYSCALE)
    artifact_templates = [template1, template2, template3]

    # Find artifact regions
    artifact_regions = find_artifact_regions(image, artifact_templates)

    # Draw rectangles around the detected regions
    if artifact_regions:
        for region in artifact_regions:
            cv2.polylines(image, [region], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Artifact Regions', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
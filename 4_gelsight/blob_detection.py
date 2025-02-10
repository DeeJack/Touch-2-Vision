"""
    Uses blob detection to find the circular artifacts, track them, and then extract the object profiles with subtraction by removing the artifacts
"""

import cv2
import numpy as np


def process_gelsight_video_simple_blob(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    processed_frames = []
    relaxed_frame_gray = None
    previous_keypoints = None

    # Setup SimpleBlobDetector parameters (tune these!)
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.7

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect blobs (our "circles") in the current frame
        keypoints = detector.detect(frame_gray)

        current_blobs = np.array([kp.pt for kp in keypoints]) if keypoints else None

        if current_blobs is not None and len(current_blobs) > 0:
            # 2. If it's the first frame, consider it the relaxed frame
            if relaxed_frame_gray is None:
                relaxed_frame_gray = frame_gray.copy()
                previous_keypoints = current_blobs
                processed_frames.append(frame.copy())
                continue

            # 3. Try to match current blobs to previous blobs (simple matching by proximity)
            if previous_keypoints is not None and len(previous_keypoints) > 0:
                src_points = np.float32(previous_keypoints)
                dst_points = np.float32(current_blobs)

                matched_src_points = []
                matched_dst_points = []
                used_current_indices = set()

                for prev_blob in previous_keypoints:
                    best_match_index = -1
                    min_distance = float("inf")
                    for i, current_blob in enumerate(current_blobs):
                        if i not in used_current_indices:
                            distance = np.linalg.norm(prev_blob - current_blob)
                            if distance < min_distance:
                                min_distance = distance
                                best_match_index = i

                    if best_match_index != -1:
                        matched_src_points.append(prev_blob)
                        matched_dst_points.append(current_blobs[best_match_index])
                        used_current_indices.add(best_match_index)

                if len(matched_src_points) >= 3:
                    matched_src_points = np.array(matched_src_points)
                    matched_dst_points = np.array(matched_dst_points)

                    # 4. Estimate affine transformation based on the movement of matched blobs
                    M = cv2.estimateAffinePartial2D(
                        matched_src_points, matched_dst_points
                    )[0]

                    if M is not None:
                        # 5. Warp the current frame to compensate for blob movement
                        compensated_frame = cv2.warpAffine(
                            frame, M, (frame.shape[1], frame.shape[0])
                        )
                        processed_frames.append(compensated_frame)
                    else:
                        processed_frames.append(frame.copy())
                else:
                    processed_frames.append(frame.copy())
            else:
                processed_frames.append(frame.copy())

            previous_keypoints = current_blobs
        else:
            processed_frames.append(frame.copy())

        frame_with_blobs = frame.copy()
        if keypoints:
            frame_with_blobs = cv2.drawKeypoints(
                frame_gray,
                keypoints,
                frame_with_blobs,
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
        cv2.imshow("Frame with Blobs", frame_with_blobs)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return processed_frames


import cv2
import numpy as np


def process_gelsight_video_profile(video_path):
    if not processed_video:
        return None

    relaxed_frame_gray = cv2.cvtColor(processed_video[0], cv2.COLOR_BGR2GRAY)

    object_profiles = []
    for i in range(1, len(processed_video)):
        compensated_frame_gray = cv2.cvtColor(processed_video[i], cv2.COLOR_BGR2GRAY)

        # Perform background subtraction
        frame_diff = cv2.absdiff(relaxed_frame_gray, compensated_frame_gray)

        # Threshold the difference image to create a binary mask
        _, thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

        object_profiles.append(morphed)

        # cv2.imshow("Object Profile", morphed)
        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()
    return object_profiles


if __name__ == "__main__":
    video_path = "videos/20220607_133934/gelsight.mp4"
    processed_video = process_gelsight_video_simple_blob(video_path)

    if processed_video:
        object_profiles = process_gelsight_video_profile(video_path)

        if object_profiles:
            for profile in object_profiles:
                cv2.imshow("Object Profile", profile)
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

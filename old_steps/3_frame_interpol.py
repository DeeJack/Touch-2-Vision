import cv2
import os

def calculate_sparse_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Detecting good features to track in the previous frame
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None)

    # Selecting the points that were successfully tracked
    good_prev_points = prev_points[status == 1]
    good_next_points = next_points[status == 1]

    return good_prev_points, good_next_points

def warp_frame_with_flow(prev_frame, next_frame, mask):
    good_prev_points, good_next_points = calculate_sparse_optical_flow(prev_frame, next_frame)
    # Estimate affine transformation using RANSAC
    if len(good_prev_points) >= 4:
        matrix, inliers = cv2.estimateAffinePartial2D(good_prev_points, good_next_points, method=cv2.RANSAC)
        warped_frame = cv2.warpAffine(prev_frame, matrix, (prev_frame.shape[1], prev_frame.shape[0]))
        mask_warped = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]))
        # Combine the warped frame and mask
        inpainted_frame = cv2.inpaint(warped_frame, mask_warped, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    else:
        inpainted_frame = cv2.inpaint(next_frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted_frame

def inpaint_video(frames_dir, masks_dir, output_video_path):
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    mask_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fps = 30
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    prev_frame = cv2.imread(frame_files[0])
    prev_mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)

    for i in range(1, len(frame_files)):
        next_frame = cv2.imread(frame_files[i])
        next_mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        inpainted_frame = warp_frame_with_flow(prev_frame, next_frame, prev_mask)
        out.write(inpainted_frame)
        prev_frame = next_frame
        prev_mask = next_mask

    out.release()

frames_dir = 'frames'
masks_dir = 'masks'
output_video_path = 'output_inpainted_video.mp4'
inpaint_video(frames_dir, masks_dir, output_video_path)

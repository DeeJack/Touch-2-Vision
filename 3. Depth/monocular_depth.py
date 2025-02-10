"""
Apply monocular depth algorithm (MiDaS) to estimate the depth map, saving them to a folder.
"""


import torch
import cv2
import numpy as np
import os

# Load the MiDaS model
model_type = "DPT_Large"  # Could also be "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Define the transformation to preprocess the input frames
transform = midas_transforms.default_transform

def estimate_depth(cropped_frame):
    # Convert BGR to LAB color space
    lab_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2LAB)
    
    input_batch = transform(lab_frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=cropped_frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map

def crop_to_mask(frame, mask):
    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    cropped_frame = frame[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    return cropped_frame, cropped_mask, (x, y, w, h)

def analyze_depth_map(depth_map, mask):
    # Apply the mask to isolate the region of interest
    masked_depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask)
    # Extract depth values within the masked region
    depth_values = masked_depth_map[mask > 0]
    if len(depth_values) == 0:
        return None  # No depth values in the masked region

    min_depth = np.min(depth_values)
    max_depth = np.max(depth_values)
    mean_depth = np.mean(depth_values)
    profile = {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "mean_depth": mean_depth,
    }
    return profile

# Process the frames to get depth maps
frames_dir = 'inpainted_frames'
masks_dir = 'masked_images'
depth_maps_dir = 'depth_maps'
cropped_depth_maps_dir = 'cropped_depth_maps'
os.makedirs(depth_maps_dir, exist_ok=True)

frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
mask_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])

for i, (frame_file, mask_file) in enumerate(zip(frame_files, mask_files)):
    print(f"Processing frame {i} ({frame_file}, {mask_file})...")
    frame = cv2.imread(frame_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    # cropped_frame, cropped_mask, (x, y, w, h) = crop_to_mask(frame, mask)
    
    # # Only proceed if there's a valid cropped region
    # if cropped_frame.size == 0 or cropped_mask.size == 0:
    #     print(f"Frame {i}: No valid cropped region found.")
    #     continue

    depth_map = estimate_depth(frame)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    # Apply colormap to create a colored visualization of depth
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    
    cv2.imwrite(os.path.join(depth_maps_dir, f'depth_map_{i:04d}.png'), depth_map)
    
    # Ensure mask and depth map have same dimensions
    mask_resized = cv2.resize(mask, (depth_map_normalized.shape[1], depth_map_normalized.shape[0]))
    cropped_depth_map = cv2.bitwise_and(depth_map_colored, depth_map_colored, mask=mask_resized)
    
    cv2.imwrite(os.path.join(cropped_depth_maps_dir, f'depth_map_{i:04d}.png'), cropped_depth_map)

    # profile = analyze_depth_map(depth_map_normalized, cropped_mask)
    # if profile:
    #     print(f"Frame {i}: {profile}")
    # else:
    #     print(f"Frame {i}: No depth values in masked region.")

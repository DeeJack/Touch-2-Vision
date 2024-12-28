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
    input_batch = transform(cropped_frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=cropped_frame.shape[:2],
        mode="bicubic",
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
frames_dir = 'frames'
masks_dir = 'masks'
depth_maps_dir = 'depth_maps'
depth_maps_cropped_dir = 'depth_maps_cropped'
os.makedirs(depth_maps_dir, exist_ok=True)
os.makedirs(depth_maps_cropped_dir, exist_ok=True)

frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
mask_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])

for i, (frame_file, mask_file) in enumerate(zip(frame_files, mask_files)):
    print(f"Processing frame {i}...")
    frame = cv2.imread(frame_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    # Resize the mask to match the frame size
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    depth_map = estimate_depth(frame)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    # Apply a colormap to the depth map
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    
    # Ensure the depth map is a 3-channel image
    if len(depth_map_colored.shape) == 2 or depth_map_colored.shape[2] == 1:
        depth_map_colored = cv2.cvtColor(depth_map_colored, cv2.COLOR_GRAY2BGR)
    
    # Draw a square around the area marked by the mask
    x, y, w, h = cv2.boundingRect(mask)
    cv2.rectangle(depth_map_colored, (x, y), (x+w, y+h), (255, 0, 0), 2)
    print(f"Rectangle coordinates: x={x}, y={y}, w={w}, h={h}")
    
    # Save the full depth map with the rectangle
    cv2.imwrite(os.path.join(depth_maps_dir, f'depth_map_{i:04d}.png'), depth_map_colored)
    
    # Save the cropped depth map
    cropped_depth_map = depth_map_colored[y:y+h, x:x+w]
    if cropped_depth_map.size > 0:
        cv2.imwrite(os.path.join(depth_maps_cropped_dir, f'cropped_depth_map_{i:04d}.png'), cropped_depth_map)

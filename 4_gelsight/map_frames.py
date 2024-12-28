"""
    Maps the frame from the inpainted video to the corresponding frame from the GelSight video, based on the timestamps.
"""

import numpy as np
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv('.env')

VIDEO_FOLDER = os.environ.get("VIDEO_FOLDER")

# Load the timestamps
time1 = np.load(os.path.join(VIDEO_FOLDER, "time1.npy"))  # Timestamps for frames in "video.mp4"
time2 = np.load(os.path.join(VIDEO_FOLDER, "time2.npy"))  # Timestamps for frames in "gelsight.mp4"

# Save the two timestamps to a txt file
np.savetxt("time1.txt", time1)
np.savetxt("time2.txt", time2)

# Paths to the folders
inpainted_frames_folder = Path("./frames")
gelsight_frames_folder = Path("./gelsight_frames")

# Get sorted lists of inpainted frames and gelsight frames
inpainted_frames = inpainted_frames_folder.glob("*.jpg")
gelsight_frames = gelsight_frames_folder.glob("*.jpg")

inpainted_frames = sorted(inpainted_frames, key=lambda x: int(x.stem.split('_')[1].replace('.jpg', '')))
gelsight_frames = sorted(gelsight_frames, key=lambda x: int(x.stem.split('_')[1].replace('.jpg', '')))

# Extract frame indices from the inpainted frame filenames (assumes format 'frame_{count:0000}.jpg')
inpainted_indices = [
    int(frame.stem.split('_')[1]) for frame in inpainted_frames
]

# Synchronize inpainted frames with GelSight frames
frame_mapping = {}  # Mapping: {inpainted_frame: gelsight_frame}

for idx in inpainted_indices:
    t1 = time1[idx]  # Get the timestamp for the inpainted frame
    # Find the closest timestamp in time2
    closest_index = np.argmin(np.abs(time2 - t1))
    if t1 != time2[closest_index]:
        print(f"Time1: {t1}, Time2: {time2[closest_index]}")
    frame_mapping[inpainted_frames_folder / f"frame_{idx:04d}.jpg"] = gelsight_frames[closest_index]

# Print the mappings
# for inpainted_frame, gelsight_frame in frame_mapping.items():
#     print(f"{inpainted_frame} -> {gelsight_frame}")

# Optional: Save the mapping to a file
import json
with open("inpainted_frame_mapping.json", "w") as f:
    json.dump({str(k): str(v) for k, v in frame_mapping.items()}, f, indent=4)

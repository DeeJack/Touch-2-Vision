# Touch 2 Vision

The project consists of a few steps:

- [X] Isolate the hand/sensor in the frame
- [X] Use inpainting to remove it from the video
- [X] Use a monocular depth algorithm to estimate the depth of the object being pressed by the sensor
- [ ] Generate the object profile from the depth information

After doing that:

- [ ] Generate an object profile considering the point movement

More information in [presentation.pdf](https://github.com/DeeJack/Touch-2-Vision/blob/main/presentation.pdf), [guidelines.pdf](https://github.com/DeeJack/Touch-2-Vision/blob/main/guidelines.pdf).

The files to use to get to my current point are:

1. Download the video "20220607_133934" from the Touch To Go dataset;
2. Install the requirements with `pip install -r requirements.txt`
3. Use [extract_frame.py](./0.%20Preparation/extract_frames.py) to take the frames from the video;
4. Use [mediapipe_frames.py](./1.%20Tracking/mediapipe_frames.py) to track the hand, creating the masks;
5. Use [ProPainter.ipynb](./notebooks/propainter.ipynb) to inpaint the masked area;
6. Use [monocular_depth.py](./3.%20Depth/monocular_depth.py) to create the depth image.

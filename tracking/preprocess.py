import cv2
import os


def load_videos(videos_path: str):
    """
    Load videos from a directory.
    Parameters
    -------
    videos_path : str
        Path to the directory containing the videos.
    Returns
    -------
    videos : List[cv2.VideoCapture]
        List of cv2.VideoCapture objects.
    """
    if not os.path.exists(videos_path):
        raise FileNotFoundError(f"Videos path not found: {videos_path}")

    videos = []
    for video_name in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video_name)
        video = cv2.VideoCapture(video_path)
        videos.append(video)

    return videos


def preprocess_frame(frame: cv2.typing.MatLike):
    """
    Preprocess the frame by resizing it before saving.
    Parameters
    -------
    frame : cv2.typing.MatLike
        The frame to preprocess
    Returns
    -------
    resized_frame : cv2.typing.MatLike
        The preprocessed frame
    """
    resized_frame = cv2.resize(frame, (640, 480))

    return resized_frame


def extract_frames(video: cv2.VideoCapture, frames_path: str, fps: int = 15):
    """
    Extract frames from a video and save them to a directory.
    Parameters
    -------
    video : cv2.VideoCapture
        Video to extract frames from.
    frames_path : str
        Path to the directory to save the frames.
    fps : int
        Frames per second to extract.
    """
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    frame_count = 0
    fps_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        fps_count += 1

        if fps_count < fps:
            continue
        fps_count = 0

        frame_path = os.path.join(frames_path, f"{frame_count}.jpg")

        preprocessed_frame = preprocess_frame(frame)
        cv2.imwrite(frame_path, preprocessed_frame)

        frame_count += 1

    print(f"Extracted {frame_count} frames from video.")


if __name__ == "__main__":
    videos = load_videos("videos")

    first_video = videos[0]

    extract_frames(first_video, "frames")

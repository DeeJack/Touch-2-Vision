"""
    Create the masks for the estimated area covered by the sensor in the image,
    based on the detected hand landmarks. The area can be approximated as an ellipse,
    rectangle, or triangle, and the corresponding mask is saved to a folder.
"""

import mediapipe as mp
import numpy as np
import cv2
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils


def calculate_area_and_vertices(
    landmarks, image_width, image_height, shape="ellipse", scale_factor=1.0
):
    if not landmarks or len(landmarks) < 21:
        return None, None  # Invalid landmarks

    # Get relevant landmark coordinates
    index_tip = np.array([landmarks[8].x * image_width, landmarks[8].y * image_height])
    thumb_tip = np.array([landmarks[4].x * image_width, landmarks[4].y * image_height])
    index_mcp = np.array([landmarks[5].x * image_width, landmarks[5].y * image_height])
    thumb_mcp = np.array([landmarks[2].x * image_width, landmarks[2].y * image_height])

    # Calculate distance between thumb and index finger tips
    distance = np.linalg.norm(index_tip - thumb_tip) * scale_factor
    distance_mcp = np.linalg.norm(index_mcp - thumb_mcp) * scale_factor

    if shape == "ellipse":
        # Approximate as an ellipse: distance = major axis, distance_mcp = minor axis
        center = ((index_tip + thumb_tip) / 2).astype(int)
        major_axis = int(distance / 2.0)
        minor_axis = int(distance_mcp / 2.0)
        angle = (
            np.arctan2(index_tip[1] - thumb_tip[1], index_tip[0] - thumb_tip[0])
            * 180
            / np.pi
        )
        area = np.pi * major_axis * minor_axis
        vertices = (
            center,
            (major_axis, minor_axis),
            int(angle),
        )  # Return ellipse parameters
    elif shape == "rectangle":
        # Approximate as a rectangle: distance = width, distance_mcp = height
        center = ((index_tip + thumb_tip) / 2).astype(int)
        width = int(distance)
        height = int(distance_mcp)
        area = distance * distance_mcp
        # Calculate rectangle vertices
        points = np.array(
            [
                [center[0] - width / 2, center[1] - height / 2],
                [center[0] + width / 2, center[1] - height / 2],
                [center[0] + width / 2, center[1] + height / 2],
                [center[0] - width / 2, center[1] + height / 2],
            ],
            dtype=np.float32,
        )
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        vertices = np.int0(box)
    elif shape == "triangle":
        # Approximate as a triangle: distance = base, distance_mcp = height
        area = 0.5 * distance * distance_mcp
        vertices = np.array([thumb_tip, index_tip, thumb_mcp], dtype=np.int32)
    else:
        raise ValueError(
            "Invalid shape specified. Choose 'ellipse', 'rectangle', or 'triangle'."
        )

    return area, vertices


def create_and_save_mask(image, vertices, shape, output_folder, frame_count):
    """
    Creates a mask from the shape vertices and saves the masked image to a folder.

    Args:
        image: The original image (BGR format).
        vertices: The vertices of the shape.
        shape: The shape ("ellipse", "rectangle", or "triangle").
        output_folder: The folder to save the masked images.
        frame_count: A counter to generate unique filenames.
    """

    if vertices is None:
        return

    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)  # Create a black mask

    if shape == "ellipse":
        center, axes, angle = vertices
        cv2.ellipse(
            mask, center, axes, angle, 0, 360, 255, -1
        )  # Draw a white ellipse on the mask
    elif shape == "rectangle":
        cv2.drawContours(
            mask, [vertices], 0, 255, -1
        )  # Draw a white rectangle on the mask
    elif shape == "triangle":
        cv2.drawContours(
            mask, [vertices], -1, 255, -1
        )  # Draw a white triangle on the mask

    # Save the masked image
    filename = os.path.join(output_folder, f"masked_frame_{frame_count:04d}.png")
    cv2.imwrite(filename, cv2.flip(mask, 1))  # Flip horizontally with flip code 1
    print(f"Saved masked image: {filename}")


# Main loop
cap = cv2.VideoCapture("./videos/20220607_133934/video.mp4")
shape = "rectangle"
output_folder = "masked_images"  # Folder to save the masks
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks to a list
            landmarks = hand_landmarks.landmark
            area, vertices = calculate_area_and_vertices(
                landmarks, image_width, image_height, shape, scale_factor=1.2
            )

            if area is not None and vertices is not None:
                # Create and save the mask
                create_and_save_mask(image, vertices, shape, output_folder, frame_count)

                if shape == "ellipse":
                    # Draw the ellipse
                    center, axes, angle = vertices
                    cv2.ellipse(image, center, axes, angle, 0, 360, (0, 0, 255), 2)
                elif shape == "rectangle":
                    # Draw the rectangle
                    cv2.drawContours(image, [vertices], 0, (0, 0, 255), 2)
                elif shape == "triangle":
                    # Draw the triangle
                    cv2.drawContours(image, [vertices], -1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        mask = np.zeros(
            (image_height, image_width), dtype=np.uint8
        )  # Create a black mask
        filename = os.path.join(output_folder, f"masked_frame_{frame_count:04d}.png")
        cv2.imwrite(filename, mask)
    frame_count += 1

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

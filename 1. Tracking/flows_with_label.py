import numpy as np
import cv2

cap = cv2.VideoCapture("videos/video.mp4")
imgs = []
NUM_FRAMES = 100
count = 0

while cap.isOpened() and count < NUM_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    count += 1

    cv2.imshow("Frame", frame)
    if cv2.waitKey(30) == 27:
        break
cv2.destroyWindow("Frame")
cap.release()


def find_flows(imgs):
    """Finds the dense optical flows"""

    optflow_params = [0.5, 3, 15, 3, 5, 1.2, 0]
    prev = imgs[0]
    flows = []
    for img in imgs[1:]:
        flow = cv2.calcOpticalFlowFarneback(prev, img, None, *optflow_params)
        flows.append(flow)
        prev = img

    return flows


# find optical flows between images
flows = find_flows(imgs)

# display flows
h, w = imgs[0].shape[:2]
hsv = np.zeros((h, w, 3), dtype=np.uint8)
hsv[..., 1] = 255

for flow in flows:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Flow", rgb)
    k = cv2.waitKey(100) & 0xFF
    if k == ord("q"):
        break
cv2.destroyWindow("Flow")


def label_flows(flows):
    """Binarizes the flows by direction and magnitude"""

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    h, w = flows[0].shape[:2]

    labeled_flows = []
    for flow in flows:
        flow = flow.reshape(h * w, -1)
        comp, labels, centers = cv2.kmeans(flow, 2, None, criteria, 10, flags)
        n = np.sum(labels == 1)
        camera_motion_label = np.argmax([labels.size - n, n])
        labeled = np.uint8(255 * (labels.reshape(h, w) == camera_motion_label))
        labeled_flows.append(labeled)
    return labeled_flows


# binarize the flows
labeled_flows = label_flows(flows)

# display binarized flows
for labeled_flow in labeled_flows:
    cv2.imshow("Labeled Flow", labeled_flow)
    k = cv2.waitKey(100) & 0xFF
    if k == ord("q"):
        break
cv2.destroyWindow("Labeled Flow")


def find_target_in_labeled_flow(labeled_flow):

    labeled_flow = cv2.bitwise_not(labeled_flow)
    bw = 10
    h, w = labeled_flow.shape[:2]
    border_cut = labeled_flow[bw : h - bw, bw : w - bw]
    conncomp, stats = cv2.connectedComponentsWithStats(border_cut, connectivity=8)[1:3]
    target_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    img = np.zeros_like(labeled_flow)
    img[bw : h - bw, bw : w - bw] = 255 * (conncomp == target_label)
    return img


for labeled_flow, img in zip(labeled_flows, imgs[:-1]):
    target_mask = find_target_in_labeled_flow(labeled_flow)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)

    display_img = cv2.merge([img, img, img])
    # contours = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
    #     1
    # ]
    contours, _ = cv2.findContours(
        target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display_img = cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Detected Target", display_img)
    k = cv2.waitKey(100) & 0xFF
    if k == ord("q"):
        break

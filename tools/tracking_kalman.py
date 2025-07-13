import os
from glob import glob
from typing import List

import cv2
import numpy as np

from ultralytics import YOLO

# Configuration
MODEL_PATH = 'weights/yolo11x-pose.pt'
IMG_DIR = 'datasets/ur_fall/raw/fall-15-cam0/images'
SEQ_START, SEQ_END = 20, 52
IMG_SUFFIX = '_30.jpg'
SAVE_DIR = 'vis'
IMG_SIZE = 640
CONF_THRESH = 0.8
DEVICE = 'cuda:0'

# Create output dir
os.makedirs(SAVE_DIR, exist_ok=True)

# Visualization colors for keypoints
colors_bgr = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255),
    (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
    (0, 128, 128), (128, 128, 0), (128, 0, 128), (0, 0, 128),
    (0, 64, 255), (0, 255, 64), (255, 64, 0), (64, 0, 255),
]

class KalmanTracker:
    def __init__(self, dt=1, p = 1, r=1, q=0.1):
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.P = np.eye(4) * p
        self.H = np.eye(4)[:2]
        self.I = np.eye(4)
        self.R = np.eye(2) * r
        self.Q = np.eye(4) * q
        self.state = np.zeros((4, 1))

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measure):
        measure = np.array(measure).reshape(2, 1)
        y = measure - self.H @ self.state
        K = self.P @ self.H.T @ np.linalg.pinv(self.H @ self.P @ self.H.T + self.R)
        self.state = self.state + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.state

# Load YOLOv8 pose model
model = YOLO(MODEL_PATH)


for scene in glob('datasets/ur_fall/raw/*'):
    img_paths = sorted(glob(f'{scene}/images/*'), key=lambda x: int(os.path.basename(x).split('_')[0]))
    tracklets: List[KalmanTracker] = []
    initialized = np.zeros(17, dtype=bool)
    missing = np.zeros(17)
    warmup = 0
    origs = []
    km = []
    # Process each frame
    for frame_idx, img_path in enumerate(img_paths):
        raw = cv2.imread(img_path)
        kalman_img = raw.copy()

        result = model.predict(source=img_path, imgsz=IMG_SIZE, device=DEVICE, verbose=False, conf=CONF_THRESH)[0]

        if not result.keypoints:
            print(f"[WARN] No keypoints detected in frame {frame_idx}")
            continue

        keypoints = result.keypoints.xy.cpu().numpy()[0].astype(int)

        if keypoints.shape[0] != 17:
            print(f"[WARN] Unexpected number of keypoints ({keypoints.shape[0]}) in frame {frame_idx}")
            continue

        for j, (x, y) in enumerate(keypoints):
            is_missing = x <= 0 or y <= 0
            if not initialized[j]:
                tracker = KalmanTracker(p = 5000, r=1, q=0.01)
                tracker.state = np.array([[x], [y], [0], [0]])
                tracklets.append(tracker)
                initialized[j] = True
            else:
                tracklets[j].predict()

            if is_missing:
                # Use estimated position
                estimate = tracklets[j].state[:2]
                new_x, new_y = estimate.flatten().astype(int)
                missing[j] += 1
                label = 'estimated'
            else:
                tracklets[j].update((x, y))
                if warmup < 10:
                    new_x, new_y = x, y
                else:
                    new_x, new_y = tracklets[j].state[:2].flatten().astype(int)

            # Draw keypoint
            kalman_img = cv2.circle(kalman_img, (new_x, new_y), 2, colors_bgr[j % len(colors_bgr)], -1)

            # Draw original (raw) keypoint
            if not is_missing:
                raw = cv2.circle(raw, (x, y), 2, colors_bgr[j % len(colors_bgr)], -1)

        warmup += 1

        origs.append(raw)
        km.append(kalman_img)

    if len(origs):
        combined = np.vstack([np.hstack([*origs]), np.hstack([*km])])
        cv2.imwrite(f"vis/{os.path.basename(scene)}.jpg", combined)
    print("✅ Processing complete.")
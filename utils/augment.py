import json
import os
from glob import glob

import cv2
import numpy as np


def add_noise_keypoints(sequence_kpt, noise_level=0.01):
    if np.random.rand() > 0.5:
        return sequence_kpt, False
    noise = np.random.uniform(-noise_level, noise_level, sequence_kpt.shape)
    noisy_keypoints = sequence_kpt + noise
    return noisy_keypoints, True

def flip_lr(sequence_kpt, norm = True, img_size = None):
    if np.random.rand() > 0.5:
        return sequence_kpt, False
    if norm:
        sequence_kpt[..., 0] = 1 - sequence_kpt[..., 0]
    else:
        sequence_kpt[:, 0] = img_size - sequence_kpt[:, 0]
    return sequence_kpt, True

def scale(sequence_kpt, factor):
    if np.random.rand() > 0.5:
        return sequence_kpt, False
    sequence_kpt= sequence_kpt*factor
    sequence_kpt[sequence_kpt > 1] = -1
    return sequence_kpt, True

def translate(sequence_kpt, factor):
    if np.random.rand() > 0.5:
        return sequence_kpt, False
    if np.sign(factor) == 1:
        factor = abs(factor)
        sequence_kpt[..., 0] = sequence_kpt[..., 0] - factor
        sequence_kpt[sequence_kpt < 0] = -1

    else:
        factor = abs(factor)
        sequence_kpt[..., 0] = sequence_kpt[..., 0] + factor
        sequence_kpt[sequence_kpt > 1] = -1

    return sequence_kpt, True


if __name__ == '__main__':
    import json
    import os
    from glob import glob

    import cv2

    bgr_colors = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 0),     # Maroon
        (128, 128, 0),   # Olive
        (0, 128, 0),     # Dark Green
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
        (0, 0, 128),     # Navy
        (255, 165, 0),   # Orange
        (255, 192, 203), # Pink
        (245, 222, 179), # Wheat
        (210, 105, 30),  # Chocolate
        (139, 69, 19),   # SaddleBrown
        (0, 255, 127),   # Spring Green
    ]

    with open('datasets/ur_fall/raw/fall-21-cam0/skeletons.json', 'r') as f:
        skels = json.load(f)['skeletons']

    imgs = []
    for i, img_path in enumerate(sorted(glob('datasets/ur_fall/raw/fall-21-cam0/images/*.jpg'), key=lambda x: int(os.path.basename(x).split('_')[0]))):
        img = cv2.imread(img_path)

        skel = skels[i]
        # noise_skel = add_noise_keypoints(np.array(skel), noise_level=2)
        # flip_skel = flip_lr(np.array(skel), norm= False, img_size=img.shape[1])
        # img, skel = scale(img, np.array(skel), 0.5)
        img, skel = translate(img, np.array(skel), -50)
        
        for kp, ((x, y), (xn, yn)) in enumerate(zip(skel, skel)):
            # img = cv2.circle(img, (x, y), 1, bgr_colors[kp], -1)
            img = cv2.circle(img, (int(xn), int(yn)), 1, bgr_colors[kp], -1)

        cv2.imwrite(f'vis/{i}.jpg', cv2.resize(img, (img.shape[1]*2, img.shape[0]*2)))
        cv2.imwrite(f'vis/{i}.jpg', cv2.resize(img, (img.shape[1]*2, img.shape[0]*2)))

import json
import os

import cv2
import numpy as np

COLORS = {
    0: (255, 0, 0),       # Red
    1: (0, 255, 0),       # Green
    2: (0, 0, 255),       # Blue
    3: (255, 255, 0),     # Yellow
    4: (0, 255, 255),     # Cyan
    5: (255, 0, 255),     # Magenta
    6: (192, 192, 192),   # Silver
    7: (128, 128, 128),   # Gray
    8: (128, 0, 0),       # Maroon
    9: (128, 128, 0),     # Olive
    10: (0, 128, 0),      # Dark Green
    11: (128, 0, 128),    # Purple
    12: (0, 128, 128),    # Teal
    13: (0, 0, 128),      # Navy
    14: (255, 165, 0),    # Orange
    15: (255, 192, 203),  # Pink
    16: (210, 105, 30),   # Chocolate
}
EDGES = [(0, 1), (0, 2),
                (1, 3),
                (2, 4),
                (5, 6), (5, 7), (5, 11),
                (6, 8), (6, 12),
                (7, 9),
                (8, 10),
                (11, 12), (11, 13),
                (12, 14),
                (13, 15),
                (14, 16),
                ]

class Visualizer:
    def __init__(self):
        pass


    def __call__(self, skel_path):
        skels = np.load(skel_path)
        gts = np.load(skel_path[:-4] + '_gt.npy')

        start_frame, end_frame = np.array(os.path.basename(skel_path)[:-4].split('_'), dtype = int)

        img_root = '/'.join(skel_path.split('/')[:5]) + '/images'
        imgs = []
        blank_imgs = []
        for k, frame_id in enumerate(range(start_frame, end_frame)):
            img = cv2.imread(f'{img_root}/{frame_id}_25.jpg')
            blank_img = np.zeros_like(img)
            h, w = img.shape[:2]
            skel = skels[k]
            c = gts[k]
            skel[:, 0] *= w
            skel[:, 1] *= h
            skel = skel.astype(int)
            for index, kp in enumerate(skel):
                if kp[0] == kp[1] == 0:
                    continue
                img = cv2.circle(img, kp, 3, COLORS[index], -1)
                blank_img = cv2.circle(blank_img, kp, 3, COLORS[index], -1)

            for i, j in EDGES:
                x1, y1 = skel[i]
                x2, y2 = skel[j]
                if x1==y1==-1 or x2==y2==-1:
                    continue
                img = cv2.line(img, (x1, y1), (x2, y2), COLORS[index], 1)
                blank_img = cv2.line(blank_img, (x1, y1), (x2, y2), COLORS[index], 1)

            # img = cv2.putText(img, f'{frame_id}/{"fall" if c > 0 else "nofall"}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            # blank_img = cv2.putText(blank_img, f'{frame_id}/{"fall" if c > 0 else "nofall"}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            
            cv2.imwrite(f"vis/{k}_skel.jpg", blank_img)

            imgs.append(img)
            blank_imgs.append(blank_img)

        combined = np.hstack([*imgs])
        combined_blank = np.hstack([*blank_imgs])
        # cv2.imwrite(f"img.jpg", combined)

        cv2.imwrite(f"test.jpg", combined_blank)

if __name__ == "__main__":

    visualizer = Visualizer()
    visualizer('datasets/le2i/data/Coffee_room_01/video (2)/sequences_kalman/223_255.npy')

    # with open('datasets/ur_fall/raw/fall-28-cam0/skeletons.json', 'r') as f:
    #     raw_skeleton = json.load(f)['skeletons']

    # with open('datasets/ur_fall/raw/fall-28-cam0/skeletons_kalman.json', 'r') as f:
    #     km_skeleton = json.load(f)['skeletons']

    # video_path = 'datasets/ur_fall/raw/fall-28-cam0'
    # img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
    

    # kms = []
    # raws = []
    # for idx, img_path in enumerate(img_paths):
    #     img = cv2.imread(img_path)

    #     km_img = img.copy()

    #     for index, kp in enumerate(raw_skeleton[idx]):
    #         if kp[0] == kp[1] == 0:
    #             continue
    #         img = cv2.circle(img, kp, 3, colors[index], -1)

    #     for index, (i, j) in enumerate(skeletons):
    #         x1, y1 = raw_skeleton[idx][i]
    #         x2, y2 = raw_skeleton[idx][j]
    #         if x1==y1==0 or x2==y2==0:
    #             continue
    #         img = cv2.line(img, (x1, y1), (x2, y2), colors[index], 1)


    #     for index, kp in enumerate(km_skeleton[idx]):
    #         if kp[0] == kp[1] == 0:
    #             continue
    #         km_img = cv2.circle(km_img, kp, 3, colors[index], -1)

    #     for index, (i, j) in enumerate(skeletons):
    #         x1, y1 = km_skeleton[idx][i]
    #         x2, y2 = km_skeleton[idx][j]
    #         if x1==y1==0 or x2==y2==0:
    #             continue
    #         km_img = cv2.line(km_img, (x1, y1), (x2, y2), colors[index], 1)


    #     img = cv2.putText(img, str(idx), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)


    #     kms.append(km_img)
    #     raws.append(img)


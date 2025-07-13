import json

import cv2

from ultralytics import YOLO

colors = {
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
skeletons = [(0, 1), (0, 2),
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

class SkeletonExtractor:
    def __init__(self, model_path: str, img_size: int = 640):
        self.model = YOLO(model_path, task = 'pose')
        self.img_size = img_size
        
    def extract_skeleton(self, image_path: str, z):
        result = self.model.predict(source=image_path, 
                                     imgsz=self.img_size, 
                                     device = 'cuda:1',
                                     verbose = False)[0]
        vis = cv2.imread(img_path)

        kps = result.keypoints.xy.cpu().numpy().astype(int)[0]

        for index, kp in enumerate(kps):
            if kp[0] == kp[1] == 0:
                continue
            vis = cv2.circle(vis, kp, 3, colors[index], -1)
        
        for index, (i, j) in enumerate(skeletons):
            x1, y1 = kps[i]
            x2, y2 = kps[j]
            if x1==y1==0 or x2==y2==0:
                continue
            vis = cv2.line(vis, (x1, y1), (x2, y2), colors[index], 1)
        cv2.imwrite(f'paper_assets/skeletons/{z}.jpg', vis)

if __name__ == "__main__":
    import os
    from glob import glob

    from tqdm import tqdm

    dataset = 'le2i'
    model_path = 'weights/yolo11m-pose.pt'  # Replace with your model path

    extractor = SkeletonExtractor(model_path=model_path)
    if dataset == 'le2i':
        video_path = "/media/ssd220/ty/fall_detection_data/le2i/data/Coffee_room_01/video (2)"
        img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
        for z, img_path in enumerate(tqdm(img_paths)):
            extractor.extract_skeleton(img_path, z)

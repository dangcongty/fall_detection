import json
import math
import os

import cv2
import numpy as np
import torch
import yaml
from kalman import KalmanTracker

from ultralytics import YOLO


class SkeletonExtractor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.get_config()
        self.init_det_model()
        self.init_kalman()

    def get_config(self):
        self.preprocess_config = self.config['data']['preprocess']
        self.detect_model_config = self.preprocess_config['detect_model']
        self.kalman_config = self.preprocess_config['kalman']
        self.dataset = self.preprocess_config['dataset']
        self.data_root = self.preprocess_config['data_root']
        self.device = self.config['device']
    
    def init_det_model(self):
        self.model = YOLO(self.detect_model_config['ckpt'], task = 'pose')
        self.img_size = self.detect_model_config['img_size']

    def init_kalman(self):
        self.initialized = np.zeros(17, dtype=bool)
        self.missing = np.zeros(17)

        self.tracklets = [KalmanTracker(p = 5000, r=1, q=0.01) for i in range(17)]
        self.old_state = [[] for i in range(17)]

        self.num_missing = self.kalman_config['num_missing']
        self.colors_bgr = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
            (255, 255, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255),
            (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
            (0, 128, 128), (128, 128, 0), (128, 0, 128), (0, 0, 128),
            (0, 64, 255), (0, 255, 64), (255, 64, 0), (64, 0, 255),
        ]

    @staticmethod
    def intersect_line_circle(h, k, r, m, b, DISTANCE_CHANGE_MAX):
        """
        h, k: Center of the circle
        r: Radius of the circle
        m, b: Slope and y-intercept of the line (y = mx + b)
        Returns: List of intersection points (x, y)

        đường tròn: (x - xc)^2 + (y - yc)^2 = R^2
        đường thẳng: y = mx + b
        """

        if m != np.inf:
            # Coefficients for the quadratic equation: Ax^2 + Bx + C = 0
            A = 1 + m**2
            B = 2 * (m * (b - k) - h)
            C = h**2 + (b - k)**2 - r**2
        else:
            A = 1
            B = -2*b
            C = b**2 - DISTANCE_CHANGE_MAX ** 2

        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            return []  # No intersection
        elif discriminant == 0:
            x = -B / (2 * A)
            y = m * x + b
            return [(x, y)]  # One point (tangent)
        else:
            sqrt_disc = math.sqrt(discriminant)
            if m != np.inf:
                x1 = (-B + sqrt_disc) / (2 * A)
                y1 = m * x1 + b
                x2 = (-B - sqrt_disc) / (2 * A)
                y2 = m * x2 + b
            else:
                x1 = x2 = b
                y1 = (-B + sqrt_disc) / (2 * A)
                y2 = (-B - sqrt_disc) / (2 * A)


            return [(x1, y1), (x2, y2)]  # Two points of intersection

    def limited_pt_change(self, pt, opt):
        def dist2points(pt1, pt2):
            return np.linalg.norm((np.array(pt1) -  np.array(pt2)))

        x, y = pt
        ox, oy = opt
        distance = dist2points([float(x), float(y)], [ox, oy]) # Nằm ngoài đường tròn
        if distance > self.kalman_config['limited_distance']['dist']:
                    # điểm dịch chuyển quá xa => có thể bị nhiễu
                    # Phương trình đường tròn (x - xc)^2 + (y - yc)^2 = R^2
                    # phương trình đường thẳng y = mx + b
            if x == ox:
                pts = self.intersect_line_circle(ox, oy, self.kalman_config['limited_distance']['dist'], np.inf, oy)
                
            else:
                m = (y-oy)/(x-ox) 
                b = y - m*x
                # Giao điểm giữa đường thẳng và đường tròn
                pts = self.intersect_line_circle(ox, oy, self.kalman_config['limited_distance']['dist'], m, b)


            if len(pts) == 2:
                pt1, pt2 = pts
                dist1 = dist2points(pt1, [x, y])
                dist2 = dist2points(pt2, [x, y])
                if dist1 < dist2:
                    intercept_pt = pt1
                else:
                    intercept_pt = pt2
            elif len(pts) == 1:
                intercept_pt = pts[0]
            else:
                print('no intercept line - circle')

            new_x, new_y = intercept_pt
        else:
            new_x, new_y = x, y

        return round(float(new_x)), round(float(new_y))


    def extract_skeleton(self, image_path: str, data: dict):
        img = cv2.imread(image_path)
        kalman_img = img.copy()

        h, w = img.shape[:2]
        result = self.model.predict(source=image_path, 
                                     imgsz=self.img_size, 
                                     device = 'cuda:1',
                                     verbose = False)[0]

        skeletons_norm = []
        skeletons = []
        if len(result):
            for j, (pt_norm, pt, kconf) in enumerate(zip(result.keypoints.xyn[0].cpu().numpy().astype(int), result.keypoints.xy[0].cpu().numpy().astype(int), result.keypoints.conf[0].cpu().numpy())):
                x, y = pt
                is_missing = x <= 0 or y <= 0
                
                if not is_missing:
                    self.tracklets[j].missing = 0
                    if not self.initialized[j]:
                        self.tracklets[j].state = np.array([[x], [y], [0], [0]])
                        self.initialized[j] = True
                        new_x, new_y = x, y
                        self.old_state[j] = [new_x, new_y]
                    else:
                        if self.kalman_config['limited_distance']['use']:
                            _x, _y = self.limited_pt_change([x, y], self.old_state[j]) # limited moving distance of x, y
                            raise "Undeveloping: limited_pt_change"
                        self.tracklets[j].predict()
                        self.tracklets[j].update((x, y), kconf)
                        new_x, new_y = self.tracklets[j].state[:2].flatten().astype(int)
                        self.old_state[j] = [new_x, new_y]
                    
                else:
                    self.tracklets[j].missing += 1
                    if self.tracklets[j].missing > self.num_missing:
                        # Del track
                        self.tracklets[j] = KalmanTracker(p = 5000, r=1, q=0.01)
                        self.initialized[j] = False
                        new_x, new_y = -1, -1

                    elif self.initialized[j]:
                        # Use estimated position
                        self.tracklets[j].predict()
                        estimate = self.tracklets[j].state[:2]
                        if self.kalman_config['limited_distance']['use']:
                            _x, _y = self.limited_pt_change([x, y], self.old_state[j]) # limited moving distance of x, y
                            raise "Undeveloping: limited_pt_change"
                        new_x, new_y = estimate
                        self.old_state[j] = [new_x, new_y]
                        
                    else:
                        new_x, new_y = -1, -1
                
                new_x, new_y = int(new_x), int(new_y)
                pt_norm = [new_x/w, new_y/h]
                pt = [new_x, new_y]
                skeletons_norm.append(pt_norm)
                skeletons.append(pt)
    
                kalman_img = cv2.circle(kalman_img, (new_x, new_y), 2, self.colors_bgr[j % len(self.colors_bgr)], -1)
                if not is_missing:
                    img = cv2.circle(img, (x, y), 2, self.colors_bgr[j % len(self.colors_bgr)], -1)

        else:
            for _ in range(17):
                skeletons_norm.append([-1, -1])
                skeletons.append([-1, -1])

        data['skeletons_norm'].append(skeletons_norm)
        data['skeletons'].append(skeletons)
        return skeletons, data, [img, kalman_img]
    


if __name__ == "__main__":
    import os
    from glob import glob

    from tqdm import tqdm

    
    dataset = 'le2i'
    if dataset in ['urfall', 'le2i']:
        model_path = 'weights/yolo11x-pose.pt'  # Replace with your model path

        if dataset == 'le2i':
            for video_path in tqdm(glob('datasets/le2i/data/*/*')):
                extractor = SkeletonExtractor('configs/base_config.yaml')
                data = {
                        'video_path': video_path,
                        'skeletons_norm': [],
                        'skeletons': []
                    }
                img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
                for img_path in tqdm(img_paths):
                    skeletons, data, [img, kalman_img] = extractor.extract_skeleton(img_path, data)
                with open(f'{video_path}/skeletons_kalman.json', 'w') as f:
                    json.dump(data, f)

        elif dataset == 'urfall':
            for video_path in tqdm(glob('datasets/ur_fall/raw/*')):
                extractor = SkeletonExtractor('configs/base_config.yaml')
                data = {
                        'video_path': video_path,
                        'skeletons_norm': [],
                        'skeletons': []
                    }
                img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
                for img_path in tqdm(img_paths):
                    skeletons, data, [img, kalman_img] = extractor.extract_skeleton(img_path, data)
                with open(f'{video_path}/skeletons_kalman.json', 'w') as f:
                    json.dump(data, f)


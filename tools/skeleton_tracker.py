import json
import os

import cv2
import numpy as np
import torch
from kalman import Kalman

from ultralytics import YOLO

bgr_colors = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (192, 192, 192), # Silver
    (128, 128, 128), # Gray
    (128, 0, 0),     # Maroon
    (128, 128, 0),   # Olive
    (0, 128, 0),     # Dark Green
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (0, 0, 128),     # Navy
    (255, 165, 0),   # Orange
    (255, 192, 203), # Pink
    (210, 105, 30),  # Chocolate
    (139, 69, 19),   # Saddle Brown
    (70, 130, 180),  # Steel Blue
    (154, 205, 50),  # Yellow Green
]

class MultipleSkeletonExtractor:
    def __init__(self, model_path: str, img_size: int = 640, max_person: int = 5):
        self.model = YOLO(model_path, task = 'pose')
        self.img_size = img_size
        self.iou_thresh = 0.6
        self.trackets = []
        self.t = 0
        
    def extract_skeleton(self, image_path: str, data: dict, c, vis_dir):
        has_person = False
        img = cv2.imread(image_path)
        h,w = img.shape[:2]
        results = self.model.predict(source=image_path, 
                                     imgsz=self.img_size, 
                                     device = 'cuda:0',
                                     verbose = False,
                                     conf = 0.1)[0]
        skeletons_norm = []
        skeletons = []
        if len(results):
            if c == 0 or not hasattr(self, 'tracker_boxes'):
                
                area = []
                area_res = []
                for result in results:
                    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0]
                    if (x2-x1)*(y2-y1) < 10000:
                        continue
                    area.append((x2-x1)*(y2-y1))
                    area_res.append(result)
                
                if len(area):
                    has_person = True
                    self.tracker_boxes = []
                    max_indice = np.argmax(area)
                    result = area_res[max_indice]
                    self.tracker_boxes.append(result.boxes.xyxy.cpu().numpy()[0])
                    for j, (pt_norm, pt, conf) in enumerate(zip(result.keypoints.xyn[0].cpu().numpy(), result.keypoints.xy[0].cpu().numpy(), result.keypoints.conf[0].cpu().numpy())):
                        kalman = Kalman(method='xy-xyv')
                        
                        measure = pt.reshape((2, 1))
                        if not (measure[0] == 0 and measure[1] == 0):
                            kalman.predict()
                            kalman.update(measure)

                        new_x, new_y = measure # t = 0 tin tưởng vào đo lường hơn
                        new_x_norm = float(new_x)/w
                        new_y_norm = float(new_y)/h
                        
                        skeletons_norm.append([new_x_norm, new_y_norm])
                        skeletons.append([int(new_x), int(new_y)])
                        self.trackets.append(kalman)


                        img = cv2.circle(img, (int(new_x), int(new_y)), 2, bgr_colors[j], -1)

                    # result.save(f'{vis_dir}/{os.path.basename(image_path)[:-4]}.jpg')
                    # img = cv2.imread(image_path)
                    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().astype(int)[0]
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.imwrite(f'{vis_dir}/{os.path.basename(image_path)[:-4]}.jpg', img)
            else:
                tx1, ty1, tx2, ty2 = self.tracker_boxes[-1]
                match_iou = []
                result_filters = []
                for k, result in enumerate(results):
                    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0]
                    center = ((x2+x1)//2, (y2+y1)//2)

                    # check if go outside
                    if x1 < 5 or x2 > (img.shape[1]-5):
                        outside = True
                    else: outside = False

                    xmin = max(x1, tx1)
                    ymin = max(y1, ty1)
                    xmax = min(x2, tx2)
                    ymax = min(y2, ty2)
                    
                    intersect = max(0, xmax - xmin) * max(0, ymax - ymin)
                    overlap = (x2-x1)*(y2-y1) + (tx2-tx1)*(ty2-ty1) - intersect
                    if overlap == 0:
                        continue
                    else:
                        iou = intersect/overlap

                    thresh = 0.1
                    if iou > thresh:
                        match_iou.append(iou)
                        result_filters.append(result)
                    
                if len(match_iou) >= 1:
                    has_person = True              
                    max_indice = np.argmax(match_iou)
                    max_iou = match_iou[max_indice]
                    result = result_filters[max_indice]
                    # self.tracker_boxes.append(result.boxes.xyxy.cpu().numpy()[0])
                    if max_iou < 0.5:
                        x1, y1, x2, y2 = self.tracker_boxes[-1]
                        self.tracker_boxes.append([x1, y1, x2, y2])
                    else:
                        x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().astype(int)[0]
                        self.tracker_boxes.append([x1, y1, x2, y2])
                    for j, (pt_norm, pt, conf) in enumerate(zip(result.keypoints.xyn[0].cpu().numpy(), result.keypoints.xy[0].cpu().numpy(), result.keypoints.conf[0].cpu().numpy())):
                        kalman = self.trackets[j]
                        kalman: Kalman
                        
                        measure = pt.reshape((2, 1))
                        if not (measure[0] == 0 and measure[1] == 0): # có joint
                            kalman.predict()
                            estimate = kalman.update(measure)
                            kalman.missing_time = 0
                        elif self.trackets[j].state.sum() != 0 and kalman.missing_time < 5: # bị mất joint
                            estimate = kalman.predict()
                            kalman.missing_time += 1
                        else:
                            estimate = measure

                        if c < 10:
                            new_x, new_y = measure
                        elif self.trackets[j].state.sum() == 0:
                            new_x, new_y = measure
                        else:
                            new_x, new_y = estimate[:2] if len(estimate) == 4 else estimate # t > 10 tin tưởng vào dự đoán hơn
                            if not ((x1 < new_x < x2) and (y1 < new_y < y2)):
                                new_x, new_y = 0, 0
                                

                        new_x_norm = float(new_x)/w
                        new_y_norm = float(new_y)/h
                        
                        skeletons_norm.append([new_x_norm, new_y_norm])
                        skeletons.append([int(new_x), int(new_y)])
                        
                        img = cv2.circle(img, (int(new_x), int(new_y)), 2, bgr_colors[j], -1)

                    
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.imwrite(f'{vis_dir}/{os.path.basename(image_path)[:-4]}.jpg', img)

                else:
                    self.missing += 1
                    for _ in range(17):
                        skeletons_norm.append([-1, -1])
                        skeletons.append([-1, -1])
        else:
            self.missing += 1
            for _ in range(17):
                skeletons_norm.append([-1, -1])
                skeletons.append([-1, -1])

        data['skeletons_norm'].append(skeletons_norm)
        data['skeletons'].append(skeletons)
        data['has_person'].append(has_person)
        return skeletons, skeletons_norm, data


if __name__ == "__main__":
    import os
    from glob import glob

    from tqdm import tqdm

    model_path = 'weights/yolo11x-pose.pt'  # Replace with your model path
    extractor = MultipleSkeletonExtractor(model_path=model_path)
    with open('datasets/UPfall/data.txt', 'w') as fdata:
        for video_path in tqdm(glob('datasets/UPfall/*')+glob('/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall/*')):
        # for video_path in a:
            if '.txt' in video_path or 'Tagged_TimeStamps' in video_path:
                continue
            video_path = 'datasets/UPfall/Subject4Activity1Trial2'
            vis_dir = f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/vis/{os.path.basename(video_path)}'
            os.makedirs(vis_dir, exist_ok=True)
            data = {
                    'video_path': video_path,
                    'skeletons_norm': [],
                    'skeletons': [],
                    'has_person': []
                }
            img_paths = sorted(glob(f'{video_path}/*.png'), key=lambda x: float(x.split('/')[-1].split('_')[-1][:-4]) + float(x.split('/')[-1].split('_')[-2])*60 + float(x.split('/')[-1].split('_')[0].split('T')[-1])*60*60)
            extractor.missing = 0
            for c, img_path in enumerate(tqdm(img_paths)):
                skeletons, skeletons_norm, data = extractor.extract_skeleton(img_path, data, c, vis_dir)
            if extractor.missing > 0.5*len(img_paths):
                fdata.write(f'{video_path}\n')
            print(extractor.missing)
            with open(f'{video_path}/skeletons.json', 'w') as f:
                json.dump(data, f)
            
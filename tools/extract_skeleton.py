import json
import os

import cv2
import numpy as np
import torch

from ultralytics import YOLO


class SkeletonExtractor:
    def __init__(self, model_path: str, img_size: int = 640):
        self.model = YOLO(model_path, task = 'pose')
        self.img_size = img_size
        
    def extract_skeleton(self, image_path: str, data: dict):
        result = self.model.predict(source=image_path, 
                                     imgsz=self.img_size, 
                                     device = 'cuda:1',
                                     verbose = False)[0]
        skeletons_norm = []
        skeletons = []
        if len(result):
            for pt_norm, pt, conf in zip(result.keypoints.xyn[0], result.keypoints.xy[0], result.keypoints.conf[0]):
                if conf < 0.5:
                    skeletons_norm.append([-1, -1])
                    skeletons.append([-1, -1])
                else:
                    skeletons_norm.append(pt_norm.cpu().numpy().tolist())
                    skeletons.append(pt.cpu().numpy().astype(int).tolist())
        else:
            for _ in range(17):
                skeletons_norm.append([-1, -1])
                skeletons.append([-1, -1])
        data['skeletons_norm'].append(skeletons_norm)
        data['skeletons'].append(skeletons)
        return skeletons, data
    
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

class MultipleSkeletonExtractor:
    def __init__(self, model_path: str, img_size: int = 640, max_person: int = 5):
        self.model = YOLO(model_path, task = 'pose')
        self.img_size = img_size
        self.iou_thresh = 0.6
        
    def extract_skeleton(self, image_path: str, data: dict, c, vis_dir):
        has_person = False
        img = cv2.imread(image_path)
        conf = 0.8 if c == 0 else 0.1
        if 'Activity11' in image_path:
            conf = 0.1
        results = self.model.predict(source=image_path, 
                                     imgsz=self.img_size, 
                                     device = 'cuda:0',
                                     verbose = False,
                                     conf = conf)[0]
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
                    self.tracker_skels = []
                    max_indice = np.argmax(area)
                    result = area_res[max_indice]
                    self.tracker_boxes.append(result.boxes.xyxy.cpu().numpy()[0])
                    self.tracker_skels.append(result.keypoints.xy.cpu().numpy()[0])
                    for pt_norm, pt, conf in zip(result.keypoints.xyn[0], result.keypoints.xy[0], result.keypoints.conf[0]):
                        if conf < 0.1:
                            skeletons_norm.append([-1, -1])
                            skeletons.append([-1, -1])
                        else:
                            skeletons_norm.append(pt_norm.cpu().numpy().tolist())
                            skeletons.append(pt.cpu().numpy().astype(int).tolist())
                    result.save(f'{vis_dir}/{os.path.basename(image_path)[:-4]}.jpg')
                    # img = cv2.imread(image_path)
                    # x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().astype(int)[0]
                    # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.imwrite(f'{vis_dir}/{os.path.basename(image_path)[:-4]}.jpg', img)
            else:
                tx1, ty1, tx2, ty2 = self.tracker_boxes[-1]
                track_center = ((tx2+tx1)//2, (ty2+ty1)//2)
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

                    thresh = 0.15
                    if iou > thresh:
                        match_iou.append(iou)
                        result_filters.append(result)
                    
                if len(match_iou) >= 1:
                    has_person = True              
                    max_indice = np.argmax(match_iou)
                    result = result_filters[max_indice]
                    self.tracker_boxes.append(result.boxes.xyxy.cpu().numpy()[0])
                    for pt_norm, pt, conf in zip(result.keypoints.xyn[0], result.keypoints.xy[0], result.keypoints.conf[0]):
                        if conf < 0.1:
                            skeletons_norm.append([-1, -1])
                            skeletons.append([-1, -1])
                        else:
                            skeletons_norm.append(pt_norm.cpu().numpy().tolist())
                            skeletons.append(pt.cpu().numpy().astype(int).tolist())
                    result.save(f'{vis_dir}/{os.path.basename(image_path)[:-4]}.jpg')

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

    
    dataset = 'upfall'
    if dataset in ['urfall', 'le2i']:
        model_path = 'weights/yolo11x-pose.pt'  # Replace with your model path

        extractor = SkeletonExtractor(model_path=model_path)
        if dataset == 'le2i':
            for video_path in tqdm(glob('datasets/le2i/data/*/*')):
                data = {
                        'video_path': video_path,
                        'skeletons_norm': [],
                        'skeletons': []
                    }
                img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
                for img_path in tqdm(img_paths):
                    skeletons, data = extractor.extract_skeleton(img_path, data)
                with open(f'{video_path}/skeletons.json', 'w') as f:
                    json.dump(data, f)

        elif dataset == 'urfall':
            for video_path in tqdm(glob('datasets/ur_fall/raw/*')):
                data = {
                        'video_path': video_path,
                        'skeletons_norm': [],
                        'skeletons': []
                    }
                img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
                for img_path in tqdm(img_paths):
                    skeletons, data = extractor.extract_skeleton(img_path, data)
                with open(f'{video_path}/skeletons.json', 'w') as f:
                    json.dump(data, f)

    elif dataset == 'hqsfd':
        model_path = 'weights/yolo11m-pose.pt'  # Replace with your model path
        extractor = MultipleSkeletonExtractor(model_path=model_path)
        for video_path in tqdm(glob('datasets/high_quality_dataset/Fall_Simulation_Data/raw/*')):
            video_path = 'datasets/high_quality_dataset/Fall_Simulation_Data/raw/Fall1_Cam2'
            data = {
                    'video_path': video_path,
                    'id': [],
                    'skeletons_norm': [],
                    'skeletons': []
                }
            img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            for c, img_path in enumerate(tqdm(img_paths)):
                if c < 1200:
                    continue
                data = extractor.extract_skeleton(img_path, data)
            with open(f'{video_path}/skeletons.json', 'w') as f:
                json.dump(data, f)

    elif dataset == 'upfall':
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
                
            
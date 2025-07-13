import os
import subprocess
import time
from glob import glob

import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
from pyinstrument import Profiler

from models.st_gcn import Model
from tools.extract_skeleton import MultipleSkeletonExtractor
from ultralytics import YOLO


def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    )
    return int(result.decode().strip().split('\n')[0])  # in MB

class Inference:
    def __init__(self, model_path, device, img_size = 640, sequence_length = 32, threshold = 0.2):

        self.device = device
        self.fall_model = Model(
            in_channels=2,
            num_class=1,
            graph_args=dict(layout='coco', strategy='spatial', max_hop=1),
            edge_importance_weighting=False,
            dropout=0.5).to(device)
        self.fall_model.load_state_dict(torch.load(model_path)['model'])
        self.fall_model.eval()
        self.img_size = img_size
        self.sequence_length = sequence_length

        self.skel_model = YOLO('weights/yolo11m-pose.pt', task = 'pose')
        self.skel_model.predict(source=np.zeros((320, 240, 3)), 
                                                imgsz=self.img_size, 
                                                device = 'cuda:1',
                                                verbose = False)[0]
        self.threshold = threshold

    def __call__(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        result_vid = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MP4V"), fps, (frame_width//2, frame_height))

        sequence_skels = np.zeros((self.sequence_length, 17, 2))
        c = 0
        t = time.time()
        before = memory_usage_mb()
        before_gpu = get_gpu_memory()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, frame.shape[1]//2:]

            result = self.skel_model.predict(source=frame, 
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

            if c < self.sequence_length:
                sequence_skels[c] = skeletons_norm
            else:
                sequence_skels[:-1] = sequence_skels[1:]
                sequence_skels[-1] = skeletons_norm

            if c >= self.sequence_length-1:
                input_sequence_skels = torch.from_numpy(sequence_skels).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(self.device).to(torch.float)
                predict = self.fall_model(input_sequence_skels).squeeze()
                pred_timestamp = predict > self.threshold

                gt = '-'
                if pred_timestamp[:-5].sum() > 3:
                    frame = cv2.putText(frame, f'fall/{gt}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    frame = cv2.putText(frame, f'nofall/{gt}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                result_vid.write(frame)
                
            
            c += 1
            print(f'{c}/{total_frames}')
        
        after = memory_usage_mb()
        
        print(fps*((time.time() - t)/c))
        print(f"[MEMORY] used {after - before:.2f} MB")
        cap.release()
        result_vid.release()
        
    def infer_images(self, paths):
        extractor = MultipleSkeletonExtractor('weights/yolo11x-pose.pt')
        frame = cv2.imread(paths[0])
        frame_height, frame_width = frame.shape[:2]
        sequence_skels = np.zeros((self.sequence_length, 17, 2))
        fps = 15
        result_vid = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MP4V"), fps, (frame_width, frame_height))
        c = 0
        data = {
                'video_path': paths[0],
                'skeletons_norm': [],
                'skeletons': []
            }
        for path in paths:
            
            frame = cv2.imread(path)
            skeletons, skeletons_norm, data = extractor.extract_skeleton(path, data, c, 'vis')

            if c < self.sequence_length:
                sequence_skels[c] = skeletons_norm
            else:
                sequence_skels[:-1] = sequence_skels[1:]
                sequence_skels[-1] = skeletons_norm

            if c >= self.sequence_length-1:
                input_sequence_skels = torch.from_numpy(sequence_skels).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(self.device).to(torch.float)
                predict = self.fall_model(input_sequence_skels).squeeze()
                pred_timestamp = predict > self.threshold

                gt = '-'
                if pred_timestamp[:-5].sum() > 3:
                    frame = cv2.putText(frame, f'fall/{gt}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    frame = cv2.putText(frame, f'nofall/{gt}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            for pt in skeletons:
                frame = cv2.circle(frame, pt, 3, (0, 255, 0), -1)
            result_vid.write(frame)
                
            
            c += 1
            print(f'{c}/{len(paths)}')



before_gpu = get_gpu_memory()
inferencer = Inference(model_path='/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/checkpoints/ver44/model_epoch_260_f1.pth',
                       device='cuda:1',
                       sequence_length=32,
                       threshold=0.2)
# inferencer(video_path='datasets/ur_fall/dataset/adl-10-cam0.mp4')

img_paths = sorted(glob(f'datasets/UPfall/Subject1Activity1Trial3/*.png'), key=lambda x: float(x.split('/')[-1].split('_')[-1][:-4]) + float(x.split('/')[-1].split('_')[-2])*60 + float(x.split('/')[-1].split('_')[0].split('T')[-1])*60*60)

inferencer.infer_images(img_paths)

after_gpu = get_gpu_memory()
print(f"[MEMORY] used {after_gpu - before_gpu:.2f} MB")

import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

dataset = 'urfall'

if dataset == 'le2i': 
    video_paths = glob('datasets/le2i/raw_data/*/*/Videos/*.avi')
    
    for vid_path in tqdm(video_paths):
        os.makedirs(f'datasets/le2i/data/{vid_path.split("/")[-3]}/{vid_path.split("/")[-1][:-4]}/images', exist_ok=True)
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        c = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f'datasets/le2i/data/{vid_path.split("/")[-3]}/{vid_path.split("/")[-1][:-4]}/images/{c}_{int(fps)}.jpg', frame)
            c += 1
        cap.release()

elif dataset == 'urfall': 
    video_paths = glob('datasets/ur_fall/dataset/*cam0*')
    
    for vid_path in tqdm(video_paths):
        os.makedirs(f'datasets/ur_fall/raw/{os.path.basename(vid_path[:-4])}/images', exist_ok=True)
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        c = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            frame = frame[:, w//2:]
            cv2.imwrite(f'datasets/ur_fall/raw/{os.path.basename(vid_path[:-4])}/images/{c}_{int(fps)}.jpg', frame)
            c += 1
        cap.release()
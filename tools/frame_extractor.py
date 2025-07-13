import os
import shutil
from datetime import datetime, time, timedelta
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset = 'hqsfd'

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

elif dataset == 'hqsfd': 
    annotations = pd.read_excel('datasets/high_quality_dataset/Data_Description.xlsx')

    # for anno in tqdm(list(annotations.iterrows())[1:]):
    #     anno = anno[1]
    #     for id in anno[12:14]:
    #         vid_path = f'datasets/high_quality_dataset/Fall_Simulation_Data/Fall{anno[0]}_Cam{id}.avi'
    #         os.makedirs(f'datasets/high_quality_dataset/Fall_Simulation_Data/raw/{os.path.basename(vid_path[:-4])}/images', exist_ok=True)
    #         cap = cv2.VideoCapture(vid_path)
    #         fps = cap.get(cv2.CAP_PROP_FPS)
    #         c = 0
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #             h, w = frame.shape[:2]
    #             cv2.imwrite(f'datasets/high_quality_dataset/Fall_Simulation_Data/raw/{os.path.basename(vid_path[:-4])}/images/{c}_{int(fps)}.jpg', frame)
    #             c += 1
    #         cap.release()

    # for vid_path in tqdm(glob("/media/ssd220/ty/fall_detection_data/high_quality_dataset/ADL*")):
    #     os.makedirs(f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/raw/{os.path.basename(vid_path[:-4])}/images', exist_ok=True)
    #     cap = cv2.VideoCapture(vid_path)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     c = 0
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         h, w = frame.shape[:2]
    #         cv2.imwrite(f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/raw/{os.path.basename(vid_path[:-4])}/images/{c}_{int(fps)}.jpg', frame)
    #         c += 1
    #     cap.release()

    # remove empty person screens
    for anno in tqdm(list(annotations.iterrows())[1:]):
        anno = anno[1]
        start_time = anno[8]
        end_time = anno[10]


        pad_time = time(0, 0, 10)
        start_time = timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second)
        end_time = timedelta(hours=end_time.hour, minutes=end_time.minute, seconds=end_time.second)
        pad_time = timedelta(hours=pad_time.hour, minutes=pad_time.minute, seconds=pad_time.second)

        start_time = start_time - pad_time
        end_time = end_time + pad_time

        fps = 30
        start_frame = start_time.total_seconds() * fps
        end_frame = end_time.total_seconds() * fps

        for k, id in enumerate(anno[12:14]):
            if k == 1:
                shutil.rmtree(f'datasets/high_quality_dataset/Fall_Simulation_Data/raw/Fall{anno[0]}_Cam{id}')
            frame_paths = glob(f'datasets/high_quality_dataset/Fall_Simulation_Data/raw/Fall{anno[0]}_Cam{id}/images/*.jpg')
            for i in range(len(frame_paths)):
                img_path = f'datasets/high_quality_dataset/Fall_Simulation_Data/raw/Fall{anno[0]}_Cam{id}/images/{i}_30.jpg'
                if start_frame < i < end_frame:
                    continue
                else:
                    if os.path.exists(img_path):
                        os.remove(img_path)
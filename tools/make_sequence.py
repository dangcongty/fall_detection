import csv
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

data = 'upfall' # urfall

if data == 'le2i':
    fps = 25    
    ignore_scene = ['Lecture_room', 'Office']
    window_size = 32 
    step = 1
    for json_path in tqdm(glob(f'datasets/le2i/data/*/*/*.json')):
        with open(json_path, 'r') as f:
            data = json.load(f)
        skels = data['skeletons_norm']
        video_path = data['video_path']
        os.makedirs(f'{video_path}/squences', exist_ok=True)

        video_name = os.path.basename(video_path)
        scene = os.path.basename(os.path.dirname(video_path))
        if scene in ignore_scene:
            continue
        # scene = 'Coffee_room_01'
        # video_name = 'video (22)'
        annotation_path = f'datasets/le2i/raw_data/{scene}/{scene}/Annotation_files/{video_name}.txt'
        with open(annotation_path, 'r') as f:
            anno = f.readlines()
            start_fall = int(anno[0].strip())
            end_fall = int(anno[1].strip())
        total_frames = len(skels)

        for start in range(0, int(total_frames - window_size), int(step)):
            end = start + window_size
            if end > total_frames:
                break

            skel_window = skels[start:end]
            np.save(f'{video_path}/squences/{start}_{end}.npy', skel_window) # 64x17x2

            label = np.zeros(len(skel_window))
            
            if (start <= start_fall <= end) and (start <= end_fall <= end):
                start_fall_squence = start_fall - start
                end_fall_squence = end_fall - start
            elif (start_fall < start) and (start < end_fall <= end):
                start_fall_squence = 0
                end_fall_squence = end_fall - start
            elif (start <= start_fall < end) and (end_fall > end):
                start_fall_squence = start_fall - start
                end_fall_squence = window_size
            elif (start_fall < start) and (end_fall > end):
                start_fall_squence = 0
                end_fall_squence = window_size
            else:
                start_fall_squence = -1
                end_fall_squence = -1

            if start_fall_squence != -1 and end_fall_squence != -1:
                label[start_fall_squence:end_fall_squence] = 1

            np.save(f'{video_path}/squences/{start}_{end}_gt.npy', label) # 64x1


elif data == 'urfall':
    fps = 30    
    window_size = 32 
    step = 1

    classes = {
        -1: 'not laying',
        0: 'falling',
        1: 'laying'
    }
    columns = ['fall_name', 'frame', 'label', 'x', 'y', 'z', 'angle', 'scale', 'energy', 'position', 'confidence']

    annotations = pd.read_csv('datasets/ur_fall/urfall-cam0-falls.csv', header=None, names = columns)

    fallnames = annotations.groupby('fall_name')
    for fallname in fallnames:
        scene = fallname[0]
        anno = fallname[1]
        label = np.where(np.array(list(anno['label'])) == 0, 1, 0)
        start_fall = np.argwhere(label == 1)[0][0]
        end_fall = np.argwhere(label == 1)[-1][0]

        gt_total_frames = len(label)

        json_path = f'datasets/ur_fall/raw/{scene}-cam0/skeletons.json'

        with open(json_path, 'r') as f:
            data = json.load(f)
        skels = data['skeletons_norm']
        video_path = data['video_path']
        os.makedirs(f'{video_path}/squences', exist_ok=True)

        video_name = os.path.basename(video_path)

        total_frames = len(skels)
        if total_frames != gt_total_frames:
            print()

        for start in range(0, int(total_frames - window_size), int(step)):
            end = start + window_size
            if end > total_frames:
                break

            skel_window = skels[start:end]
            np.save(f'{video_path}/squences/{start}_{end}.npy', skel_window) # 64x17x2

            label = np.zeros(len(skel_window))
            
            if (start <= start_fall <= end) and (start <= end_fall <= end):
                start_fall_squence = start_fall - start
                end_fall_squence = end_fall - start
            elif (start_fall < start) and (start < end_fall <= end):
                start_fall_squence = 0
                end_fall_squence = end_fall - start
            elif (start <= start_fall < end) and (end_fall > end):
                start_fall_squence = start_fall - start
                end_fall_squence = window_size
            elif (start_fall < start) and (end_fall > end):
                start_fall_squence = 0
                end_fall_squence = window_size
            else:
                start_fall_squence = -1
                end_fall_squence = -1

            if start_fall_squence != -1 and end_fall_squence != -1:
                label[start_fall_squence:end_fall_squence] = 1

            np.save(f'{video_path}/squences/{start}_{end}_gt.npy', label) # 64x1


    for adl in tqdm(glob('datasets/ur_fall/raw/adl-*-cam0')):
        json_path = f'{adl}/skeletons.json'
        with open(json_path, 'r') as f:
            data = json.load(f)
        skels = data['skeletons_norm']
        video_path = data['video_path']
        os.makedirs(f'{video_path}/squences', exist_ok=True)

        total_frames = len(skels)

        for start in range(0, int(total_frames - window_size), int(step)):
            end = start + window_size
            if end > total_frames:
                break

            skel_window = skels[start:end]
            np.save(f'{video_path}/squences/{start}_{end}.npy', skel_window) # 64x17x2

            label = np.zeros(len(skel_window))
            np.save(f'{video_path}/squences/{start}_{end}_gt.npy', label) # 64x1

elif data == 'upfall':
    window_size = 32 
    step = 1
    for subject in glob(f'datasets/UPfall/Tagged_TimeStamps/*'):
        for act in glob(f'{subject}/*'):
            for trial in glob(f'{act}/*.csv'):
                name = os.path.basename(trial)[:-4]
                annotations = pd.read_csv(trial, header=None)
                fall = []
                for k, info in enumerate(annotations.iterrows()):
                    if k == 0:
                        continue
                    ti, sj, act, tr, c  = list(info[1])
                    if c in ['1', '2', '3', '4', '5']:
                        fall.append(k)


                json_path = f'datasets/UPfall/{name}/skeletons.json'
                if not os.path.exists(json_path):
                    json_path = f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall/{name}/skeletons.json'

                with open(json_path, 'r') as f:
                    data = json.load(f)
                skels = data['skeletons_norm']
                video_path = data['video_path']
                os.makedirs(f'{video_path}/squences', exist_ok=True)
                print(f'{video_path}/squences')

                total_frames = len(skels)
                if len(fall):
                    start_fall = fall[0]
                    end_fall = fall[-1]

                    for start in range(0, int(total_frames - window_size), int(step)):
                        end = start + window_size
                        if end > total_frames:
                            break

                        skel_window = skels[start:end]
                        if len(skel_window[-1]) == 0:
                            skel_window
                        np.save(f'{video_path}/squences/{start}_{end}.npy', skel_window) # 64x17x2

                        label = np.zeros(len(skel_window))
                        
                        if (start <= start_fall <= end) and (start <= end_fall <= end):
                            start_fall_squence = start_fall - start
                            end_fall_squence = end_fall - start
                        elif (start_fall < start) and (start < end_fall <= end):
                            start_fall_squence = 0
                            end_fall_squence = end_fall - start
                        elif (start <= start_fall < end) and (end_fall > end):
                            start_fall_squence = start_fall - start
                            end_fall_squence = window_size
                        elif (start_fall < start) and (end_fall > end):
                            start_fall_squence = 0
                            end_fall_squence = window_size
                        else:
                            start_fall_squence = -1
                            end_fall_squence = -1

                        if start_fall_squence != -1 and end_fall_squence != -1:
                            label[start_fall_squence:end_fall_squence] = 1

                        np.save(f'{video_path}/squences/{start}_{end}_gt.npy', label) # 64x1
                else:
                    for start in range(0, int(total_frames - window_size), int(step)):
                        end = start + window_size
                        if end > total_frames:
                            break
                        skel_window = skels[start:end]
                        np.save(f'{video_path}/squences/{start}_{end}.npy', skel_window) # 64x17x2
                        np.save(f'{video_path}/squences/{start}_{end}_gt.npy', np.zeros(len(skel_window))) # 64x1

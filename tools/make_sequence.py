import csv
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA = {
    0: 'urfall',
    1: 'le2i'
}

class PrepareData:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.threshold_filter = self.config['val']['threshold']['classify']
        
        self.dataset = int(input('Choose dataset:\n0: urfall\n1: le2i\n'))
        try:
            self.use_kalman = int(input('Use Kalman (choose 0 or 1): '))
        except:
            self.use_kalman = 1

        try:
            self.split_by = int(input('Choose split method:\n0: scene\n1: sequence\n'))
        except:
            self.split_by = 'scene'

        if self.use_kalman:
            self.skel_path = 'skeletons_kalman'
            self.sequence_path = 'sequences_kalman'
        else:
            self.sequence_path = 'sequences'
            self.skel_path = 'skeletons'

        try:
            self.window_size = int(input('Window Size (default 32): '))
        except:
            self.window_size = 32

        try:
            self.stride = int(input('Stride (default 1): '))
        except:
            self.stride = 1

        print(f'Dataset: {DATA[self.dataset]} -- Windows Size: {self.window_size} -- Stride: {self.stride} -- Use Kalman: {self.use_kalman} -- Split method: {self.split_by}')

    def normalized_gaussian(self, min_x, max_x, min_value=0.5):
        import numpy as np

        mu = (min_x + max_x) / 2
        sigma = (max_x - min_x) / 6
        x = np.linspace(min_x, max_x, max_x - min_x + 1)
        y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Scale y so that min becomes 0.5 and max stays 1
        y_min, y_max = y.min(), y.max()
        y_scaled = min_value + (y - y_min) * (1 - min_value) / (y_max - y_min)

        return x, y_scaled

    def extract_le2i(self):
        ignore_scene = ['Lecture_room', 'Office']
        metadata = {
            'fall': [],
            'nofall': []
        }
        for json_path in tqdm(glob(f'datasets/le2i/data/*/*/{self.skel_path}.json')):
            with open(json_path, 'r') as f:
                data = json.load(f)
            skels = data['skeletons_norm']
            video_path = data['video_path']
            os.makedirs(f'{video_path}/{self.sequence_path}', exist_ok=True)

            video_name = os.path.basename(video_path)
            scene = os.path.basename(os.path.dirname(video_path))
            if scene in ignore_scene:
                continue

            annotation_path = f'datasets/le2i/raw_data/{scene}/{scene}/Annotation_files/{video_name}.txt'
            with open(annotation_path, 'r') as f:
                anno = f.readlines()
                start_fall = int(anno[0].strip()) - 1
                end_fall = int(anno[1].strip()) - 1

            # metadata[scene] = {} if scene not in metadata else metadata[scene]
            # fall_duration[scene] = {} if scene not in fall_duration else fall_duration[scene]
            total_frames = len(skels)
            if start_fall != -1 and end_fall != -1:
                gaussian = self.normalized_gaussian(start_fall, end_fall)[1]
                soft_label = np.zeros(total_frames)
                soft_label[start_fall:end_fall+1] = gaussian

                metadata['fall'].append([scene, video_name, start_fall, end_fall, end_fall-start_fall])
                # fall_duration[scene][video_name] = [start_fall, end_fall, end_fall-start_fall]
            else:
                # adl scene
                soft_label = np.zeros(total_frames)
                metadata['nofall'].append([scene, video_name, 0, 0, 0])


            for start in range(0, int(total_frames - self.window_size), int(self.stride)):
                end = start + self.window_size
                if end > total_frames:
                    break

                _label = soft_label[start:end]
                skel_window = skels[start:end]
                while len(skel_window) < self.window_size:
                    skel_window.append(skel_window[-1])
                    _label.append(_label[-1])
                np.save(f'{video_path}/{self.sequence_path}/{start}_{end}.npy', skel_window) # 64x17x2
                np.save(f'{video_path}/{self.sequence_path}/{start}_{end}_gt.npy', _label) # 64x17x2

        return metadata
        
    def extract_urfall(self):
        columns = ['fall_name', 'frame', 'label', 'x', 'y', 'z', 'angle', 'scale', 'energy', 'position', 'confidence']
        annotations = pd.read_csv('datasets/ur_fall/urfall-cam0-falls.csv', header=None, names = columns)
        fallnames = annotations.groupby('fall_name')

        # fall
        fall_duration = {}
        for fallname in tqdm(fallnames):
            scene = fallname[0]
            anno = fallname[1]
            label = np.where(np.array(list(anno['label'])) == 0, 1, 0)
            start_fall = np.argwhere(label == 1)[0][0]
            end_fall = np.argwhere(label == 1)[-1][0]

            gaussian = self.normalized_gaussian(start_fall, end_fall)[1]
            soft_label = np.zeros(label.shape[0])
            soft_label[start_fall:end_fall+1] = gaussian

            gt_total_frames = len(label)
            json_path = f'datasets/ur_fall/raw/{scene}-cam0/{self.skel_path}.json'

            with open(json_path, 'r') as f:
                data = json.load(f)
            skels = data['skeletons_norm']
            video_path = data['video_path']
            os.makedirs(f'{video_path}/{self.sequence_path}', exist_ok=True)

            total_frames = len(skels)
            if total_frames != gt_total_frames:
                raise 'total_frames != gt_total_frames'

            for start in range(0, int(total_frames - self.window_size), int(self.stride)):
                end = start + self.window_size
                if end > total_frames:
                    break

                _label = soft_label[start:end]
                skel_window = skels[start:end]
                np.save(f'{video_path}/{self.sequence_path}/{start}_{end}.npy', skel_window) # 64x17x2
                np.save(f'{video_path}/{self.sequence_path}/{start}_{end}_gt.npy', _label) # 64x17x2

            fall_duration[scene] = [start_fall, end_fall, end_fall-start_fall]

        # activity daily (no fall)
        for adl in tqdm(glob('datasets/ur_fall/raw/adl-*-cam0')):
            json_path = f'{adl}/{self.skel_path}.json'
            with open(json_path, 'r') as f:
                data = json.load(f)
            skels = data['skeletons_norm']
            video_path = data['video_path']
            os.makedirs(f'{video_path}/{self.sequence_path}', exist_ok=True)

            total_frames = len(skels)
            label = np.zeros(total_frames)

            for start in range(0, int(total_frames - self.window_size), int(self.stride)):
                end = start + self.window_size
                if end > total_frames:
                    break

                skel_window = skels[start:end]
                _label = label[start:end]
                np.save(f'{video_path}/{self.sequence_path}/{start}_{end}.npy', skel_window) # 64x17x2
                np.save(f'{video_path}/{self.sequence_path}/{start}_{end}_gt.npy', _label) # 64x1
        return fall_duration
    
    def split_le2i(self, metadata):
        if self.use_kalman:
            suffix = '_kalman'
        else:
            suffix = ''

        if self.split_by == 1:
            data_paths = glob(f'datasets/ur_fall/raw/*/sequences{suffix}/*_gt.npy')
            train_paths, test_paths = train_test_split(data_paths, test_size=0.2, random_state=317207)

        else:

            fall_scene_paths = [f'datasets/le2i/data/{scene}/{vid_name}' for scene, vid_name, _, _, _ in metadata['fall']]
            adl_scene_paths = [f'datasets/le2i/data/{scene}/{vid_name}' for scene, vid_name, _, _, _ in metadata['nofall']]

            fall_duration = {}
            for scene, vid_name, _, _, duration in metadata['fall'] + metadata['nofall']:
                fall_duration[f'datasets/le2i/data/{scene}/{vid_name}'] =  duration



            train_fall_scene, test_fall_scene = train_test_split(fall_scene_paths, test_size=0.2, random_state=317207)
            train_adl_scene, test_adl_scene = train_test_split(adl_scene_paths, test_size=0.2, random_state=317207)
            train_paths = []
            test_paths = []
            for scene in train_fall_scene + train_adl_scene:
                train_paths += glob(f'{scene}/sequences{suffix}/*_gt.npy')
            for scene in test_fall_scene + test_adl_scene:
                test_paths += glob(f'{scene}/sequences{suffix}/*_gt.npy')

            # filter
            filters_train_fall, filters_test_fall = self.filter_timestamp_le2i(train_paths, test_paths, ratio = self.threshold_filter, fall_duration=fall_duration)


        with open(f'datasets/le2i/train{suffix}.txt', 'w') as f:
            for path in filters_train_fall:
                f.write(path + '\n')

        with open(f'datasets/le2i/test{suffix}.txt', 'w') as f:
            for path in filters_test_fall:
                f.write(path + '\n')

    def filter_timestamp_le2i(self, train_paths, test_paths, ratio, fall_duration):
        filters_train_fall = []
        for npy_path in train_paths:
            duration = fall_duration['/'.join(npy_path.split('/')[:-2])]
            skel = np.load(npy_path)
            skel = np.where(skel > 0, 1, 0)
            if skel.sum() != 0:
                if (skel.sum() > duration*ratio):
                    filters_train_fall.append(npy_path)
            else:
                filters_train_fall.append(npy_path)

        filters_test_fall = []
        for npy_path in test_paths:
            duration = fall_duration['/'.join(npy_path.split('/')[:-2])]
            skel = np.load(npy_path)
            skel = np.where(skel > 0, 1, 0)
            if skel.sum() != 0:
                if (skel.sum() > duration*ratio):
                    filters_test_fall.append(npy_path)
            else:
                filters_test_fall.append(npy_path)

        return filters_train_fall, filters_test_fall


    def split_urfall(self, fall_duration):
        if self.use_kalman:
            suffix = '_kalman'
        else:
            suffix = ''

        if self.split_by == 1:
            data_paths = glob(f'datasets/ur_fall/raw/*/sequences{suffix}/*_gt.npy')
            train_paths, test_paths = train_test_split(data_paths, test_size=0.2, random_state=317207)

        else:
            fall_scene_paths = glob(f'datasets/ur_fall/raw/fall*')
            adl_scene_paths = glob(f'datasets/ur_fall/raw/adl*')
            train_fall_scene, test_fall_scene = train_test_split(fall_scene_paths, test_size=0.2, random_state=317207)
            train_adl_scene, test_adl_scene = train_test_split(adl_scene_paths, test_size=0.2, random_state=317207)
            train_paths = []
            test_paths = []
            for scene in train_fall_scene + train_adl_scene:
                train_paths += glob(f'{scene}/sequences{suffix}/*_gt.npy')
            for scene in test_fall_scene + test_adl_scene:
                test_paths += glob(f'{scene}/sequences{suffix}/*_gt.npy')

            # filter
            filters_train_fall, filters_test_fall = self.filter_timestamp_urfall(train_paths, test_paths, suffix, ratio = self.threshold_filter, fall_duration=fall_duration)

        with open(f'datasets/ur_fall/train{suffix}.txt', 'w') as f:
            for path in filters_train_fall:
                f.write(path + '\n')

        with open(f'datasets/ur_fall/test{suffix}.txt', 'w') as f:
            for path in filters_test_fall:
                f.write(path + '\n')

    def filter_timestamp_urfall(self, train_paths, test_paths, suffix, ratio, fall_duration):
        filters_train_fall = []
        for npy_path in train_paths:
            if 'adl' not in npy_path:
                duration = fall_duration['-'.join(npy_path.split('/')[3].split('-')[:2])]
            skel = np.load(npy_path)
            skel = np.where(skel > 0, 1, 0)
            if skel.sum() != 0:
                if (skel.sum() > duration[-1]*ratio):
                    filters_train_fall.append(npy_path)
            else:
                filters_train_fall.append(npy_path)

        filters_test_fall = []
        for npy_path in test_paths:
            if 'adl' not in npy_path:
                duration = fall_duration['-'.join(npy_path.split('/')[3].split('-')[:2])]
            skel = np.load(npy_path)
            skel = np.where(skel > 0, 1, 0)
            if skel.sum() != 0:
                if (skel.sum() > duration[-1]*ratio):
                    filters_test_fall.append(npy_path)
            else:
                filters_test_fall.append(npy_path)
        return filters_train_fall, filters_test_fall

    def __call__(self):
        if self.dataset == 0:
            fall_duration = self.extract_urfall()
            self.split_urfall(fall_duration)

        elif self.dataset == 1:
            metadata = self.extract_le2i()
            self.split_le2i(metadata)


if __name__ == '__main__':
    preparedataset = PrepareData('configs/base_config.yaml')
    preparedataset()
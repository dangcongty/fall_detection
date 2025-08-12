import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

sys.path.append(f'{os.getcwd()}/')
sys.path.append(f'{os.getcwd()}/utils')
from augment import *


class FallDataset:
    def __init__(self, 
                 data_path, 
                 ):
        self.data_path = data_path
        
        with open(self.data_path, 'r') as f:
            self.data = f.readlines()

        paths = {}
        for path in self.data:
            vid_name = '/'.join(path.strip().split('/')[:-2])
            paths[vid_name] = [] if vid_name not in paths else paths[vid_name]
            t_start = int(path.strip().split('/')[-1].split('_')[0])
            paths[vid_name].append(t_start)

        self.reformat_paths = []
        for vid_name in paths:
            t_start, t_end = np.min(paths[vid_name]), np.max(paths[vid_name])
            self.reformat_paths.append([f'{vid_name}/sequences_kalman/{s}_{s+32}_gt.npy' for s in range(t_start, t_end+1)])

        self.vid_id = 0
    def __len__(self):
        return len(self.data)

    def __call__(self):
        # load data
        idx = 0
        for vid_id in range(len(self.reformat_paths)):

            end_sequence = int(os.path.basename(self.reformat_paths[vid_id][-1]).split('_')[1])
            last_start_sequence = int(os.path.basename(self.reformat_paths[vid_id][-1]).split('_')[0])
            pad_length = end_sequence - last_start_sequence

            for idx in range(len(self.reformat_paths[vid_id])):
                label_path = self.reformat_paths[vid_id][idx]
                skel_path = label_path.replace('_gt', '')
                skel = np.load(skel_path)
                label = np.load(label_path)
                
                # transform torch
                skel = torch.from_numpy(skel).float().permute(2, 0, 1)  # T, V, C -> C, T, V

                # if idx == len(self.reformat_paths[vid_id])-1:
                #    _skel = skel.clone()
                #    _label = label.copy()
                #    last_skel = skel[:, -1]
                #    last_label = label[-1]
                #    for p in range(pad_length):
                #        _skel[:, :-1] = _skel[:, 1:].clone()
                #        _skel[:, -1] = last_skel
                #        _label[:-1] = _label[1:]
                #        _label[-1] = last_label
                #        yield skel, label, vid_id
                # else:
                yield skel, label, vid_id

if __name__ == "__main__":
    # Example usage

    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    train_file_name = 'train.txt'
    val_file_name = 'test.txt'
    if data_config['use_kalman'] == True:
        train_file_name = 'train_kalman.txt'
        val_file_name = 'test_kalman.txt'

    train_dataset = FallDataset(data_path=f'{data_config["root"]}/{train_file_name}')


    for data in train_dataset():
        a = 1
        print(data)
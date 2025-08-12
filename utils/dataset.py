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


class FallDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 mixup_data,
                 arguments,
                 transform = False
                 ):
        self.data_path = data_path
        
        with open(self.data_path, 'r') as f:
            self.data = f.readlines()

        if mixup_data['data_name'] != '':
            with open('datasets/le2i/test.txt', 'r') as f:
                self.sub_data = f.readlines()
            self.sub_data = self.sub_data[:int(len(self.sub_data)*0.2)]

        self.add_noise = arguments['add_noise']
        self.flip_lr = arguments['flip_lr']
        self.scale = arguments['scale']
        self.translate = arguments['translate']

        self.transform = transform

        self.get_weighted_sampler()

    def get_weighted_sampler(self):
        gts = []
        for gt_path in self.data:
            gt_path = gt_path.strip()
            gts.append(1 if np.load(gt_path).sum() > 0 else 0)
        self.gts = torch.tensor(gts)

        class_sample_counts = torch.bincount(self.gts)
        weights = 1.0 / class_sample_counts
        sample_weights = weights[self.gts]
        self.sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.weight_loss = len(gts)/self.gts.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load data
        label_path = self.data[idx].strip()
        skel_path = label_path.replace('_gt', '')
        skel = np.load(skel_path)
        label = np.load(label_path)
        
        # augmentation
        mask_missing = skel<0
        if self.transform:
            skel, st1 = flip_lr(skel) if self.flip_lr else skel
            skel, st2 = scale(skel, factor=np.random.uniform(0.8, 1.2)) if self.scale else skel
            skel, st3 = translate(skel, factor=np.random.uniform(-0.02, 0.02)) if self.translate else skel
            skel, st4 = add_noise_keypoints(skel, noise_level=0.008) if self.add_noise else skel
            skel[mask_missing] = -1.0


        label_cls = np.where(label > 0, 1, 0)
        # transform torch
        label_timestamp = torch.from_numpy(label).float()
        label_cls = torch.from_numpy(label_cls).float()
        skel = torch.from_numpy(skel).float().permute(2, 0, 1)  # T, V, C -> C, T, V
        
        return skel, (label_cls, label_timestamp), label_path, label.sum() > 0

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

    train_dataset = FallDataset(data_path=f'{data_config["root"]}/{train_file_name}',
                                arguments = data_config['augmentations'],
                                mixup_data = data_config['mixup_data'],
                                transform = True)


    sample = train_dataset.__getitem__(0)
    print(sample)
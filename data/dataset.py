import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(f'{os.getcwd()}/')
sys.path.append(f'{os.getcwd()}/data')
from augment import *


class FallDataset(Dataset):
    def __init__(self, data_path, transform=True):
        self.data_path = data_path
        
        with open(self.data_path, 'r') as f:
            self.data = f.readlines()

        # with open('datasets/le2i/test.txt', 'r') as f:
        #     self.sub_data = f.readlines()
        
        # self.sub_data = self.sub_data[:int(len(self.sub_data)*0.2)]

        # self.data = self.data + self.sub_data

        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_path = self.data[idx].strip()
        skel_path = label_path.replace('_gt', '')

        skel = np.load(skel_path)
        label = np.load(label_path)
        
        if self.transform:
            mask_missing = skel==-1

            skel = add_noise_keypoints(skel, noise_level=0.01)
            skel = flip_lr(skel)
            skel = scale(skel, factor=np.random.uniform(-1.5, 1.5))
            skel = translate(skel, factor=np.random.uniform(-0.05, 0.05))
            skel[mask_missing] = -1.0

        
        label = torch.from_numpy(label).float()
        skel = torch.from_numpy(skel).float().permute(2, 0, 1)  # T, V, C -> C, T, V
        
        return skel, label, label_path, label.sum() > 0

if __name__ == "__main__":
    # Example usage
    dataset = FallDataset(data_path='datasets/ur_fall/train.txt')
    print(len(dataset))
    sample = dataset.__getitem__(0)
    print(sample)
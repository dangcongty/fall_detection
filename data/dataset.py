import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(f'{os.getcwd()}/')
sys.path.append(f'{os.getcwd()}/data')
from augment import add_noise_keypoints, interpolate_temporal


class FallDataset(Dataset):
    def __init__(self, data_path, transform=True):
        self.data_path = data_path
        
        with open(self.data_path, 'r') as f:
            self.data = f.readlines()

        self.transform = transform
        self.cell_size = 2

    def encode(self, timestamp, window_size):
        num_cells = int(window_size/self.cell_size)
        label = np.zeros((num_cells, 3))
        try:
            if len(timestamp) > 0:
                unnorm_timestamp = timestamp * window_size
                center_fullscale = unnorm_timestamp.mean()

                cell_contain_fall = int(center_fullscale/self.cell_size)

                offset_center = center_fullscale/self.cell_size - cell_contain_fall
                fall_width = timestamp[1] - timestamp[0]
                
                label[cell_contain_fall, 0] = 1
                label[cell_contain_fall, 1] = offset_center
                label[cell_contain_fall, 2] = fall_width
        except:
            print

        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_path = self.data[idx].strip()
        # label_path = 'datasets/le2i/data/Coffee_room_01/video (27)/squences/160/_224_gt.npy'
        skel_path = label_path.replace('_gt', '')

        skel = np.load(skel_path)
        label = np.load(label_path)
        
        if self.transform:
            skel = add_noise_keypoints(skel, noise_level=0.01)
            # skel = interpolate_temporal(skel, num_input_sequences=skel.shape[0], num_output_sequences=skel.shape[0]*2)
        
        label = torch.from_numpy(label).float()
        skel = torch.from_numpy(skel).float().permute(2, 0, 1)  # T, V, C -> C, T, V
        
        return skel, label, label_path

if __name__ == "__main__":
    # Example usage
    dataset = FallDataset(data_path='datasets/le2i/train.txt')
    print(len(dataset))
    sample = dataset[0]
    print(sample)
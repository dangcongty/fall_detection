import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def filter_urfall():
    filters = []
    for npy_path in tqdm(glob('datasets/ur_fall/raw/*/squences/*_gt.npy')):
        skel = np.load(npy_path)
        if skel.sum() != 0:
            if (skel.sum() > len(skel)*0.2):
                filters.append(npy_path)
        else:
            filters.append(npy_path)

    train_paths, test_paths = train_test_split(filters, test_size=0.2, random_state=317207)


    with open('datasets/ur_fall/train.txt', 'w') as f:
        for path in train_paths:
            f.write(path + '\n')

    with open('datasets/ur_fall/test.txt', 'w') as f:
        for path in test_paths:
            f.write(path + '\n')


def filter_le2i():
    filters = []
    for npy_path in tqdm(glob('datasets/le2i/data/*/*/squences/*_gt.npy')):
        skel = np.load(npy_path)
        if skel.sum() != 0:
            if (skel.sum() > len(skel)*0.2):
                filters.append(npy_path)
        else:
            filters.append(npy_path)

    train_paths, test_paths = train_test_split(filters, test_size=0.2, random_state=317207)


    with open('datasets/le2i/train.txt', 'w') as f:
        for path in train_paths:
            f.write(path + '\n')

    with open('datasets/le2i/test.txt', 'w') as f:
        for path in test_paths:
            f.write(path + '\n')

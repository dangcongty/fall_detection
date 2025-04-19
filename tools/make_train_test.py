import os
from glob import glob

from sklearn.model_selection import train_test_split

data_paths = glob('datasets/ur_fall/raw/*/squences/*_gt.npy')

train_paths, test_paths = train_test_split(data_paths, test_size=0.2, random_state=317207)

with open('datasets/ur_fall/train.txt', 'w') as f:
    for path in train_paths:
        f.write(path + '\n')

with open('datasets/ur_fall/test.txt', 'w') as f:
    for path in test_paths:
        f.write(path + '\n')
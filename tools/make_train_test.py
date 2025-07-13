import os
import re
from glob import glob

from sklearn.model_selection import train_test_split

data = 'upfall'
if data == 'ur_fall':
    data_paths = glob('datasets/ur_fall/raw/*/squences/*_gt.npy')

    train_paths, test_paths = train_test_split(data_paths, test_size=0.2, random_state=317207)

    with open('datasets/ur_fall/train.txt', 'w') as f:
        for path in train_paths:
            f.write(path + '\n')

    with open('datasets/ur_fall/test.txt', 'w') as f:
        for path in test_paths:
            f.write(path + '\n')

elif data == 'upfall':
    train_subjects = [1, 3, 4, 7, 10, 11, 12, 13, 14]
    val_subjects = [1, 3, 4]
    test_subjects = [15, 16, 17]

    train_paths = []
    val_paths = []
    test_paths = []

    for path in glob('datasets/UPfall/*/squences/*_gt.npy') + glob(f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall/*/squences/*_gt.npy'):
        match = re.search(r"Subject(\d+)Activity(\d+)Trial(\d+)", path.split('/')[-3])
        subject = int(match.group(1))
        activity = int(match.group(2))
        trial = int(match.group(3))
        if trial == 3:
            test_paths.append(path)
        else:
            if subject in val_subjects:
                if trial == 2:
                    val_paths.append(path)
                    continue
                
            train_paths.append(path)


            
    with open('datasets/UPfall/train.txt', 'w') as f:
        for path in train_paths:
            f.write(path + '\n')

    with open('datasets/UPfall/val.txt', 'w') as f:
        for path in val_paths:
            f.write(path + '\n')    

    with open('datasets/UPfall/test.txt', 'w') as f:
        for path in test_paths:
            f.write(path + '\n')    
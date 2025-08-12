import os
import subprocess

data = 'falldetection'
if data == 'falldetection':
    os.makedirs('datasets/falldetection_data', exist_ok=True)
    for train_id in [1301, 1790, 832, 722, 1378, 1392, 807, 758, 1843, 569, 1260, 489, 731, 1219, 1954, 581]:
        os.makedirs(f'datasets/falldetection_data/train/{train_id}', exist_ok=True)
        subprocess.call(['wget', f'https://falldataset.com/data/{train_id}/{train_id}.tar.gz', '-O', f'datasets/falldetection_data/train/{train_id}/{train_id}.tar.gz'])
        subprocess.call(['tar', '-xvf', f'datasets/falldetection_data/train/{train_id}/{train_id}.tar.gz', '-C', f'datasets/falldetection_data/train/{train_id}'])

    for val_id in [1176, 2123]:
        os.makedirs(f'datasets/falldetection_data/val/{val_id}', exist_ok=True)
        subprocess.call(['wget', f'https://falldataset.com/data/{val_id}/{val_id}.tar.gz', '-O', f'datasets/falldetection_data/val/{val_id}/{val_id}.tar.gz'])
        subprocess.call(['tar', '-xvf', f'datasets/falldetection_data/val/{val_id}/{val_id}.tar.gz', '-C', f'datasets/falldetection_data/val/{val_id}'])


    for test_id in [832, 786, 925]:
        os.makedirs(f'datasets/falldetection_data/test/{test_id}', exist_ok=True)
        subprocess.call(['wget', f'https://falldataset.com/data/{test_id}/{test_id}.tar.gz', '-O', f'datasets/falldetection_data/test/{test_id}/{test_id}.tar.gz'])
        subprocess.call(['tar', '-xvf', f'datasets/falldetection_data/test/{test_id}/{test_id}.tar.gz', '-C', f'datasets/falldetection_data/test/{test_id}'])

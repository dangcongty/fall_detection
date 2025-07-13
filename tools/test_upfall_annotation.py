import os
import shutil
from glob import glob

import cv2
import numpy as np

shutil.rmtree('vis')
os.makedirs('vis')

img_paths = glob('datasets/UPfall/Subject4Activity1Trial2/*.png')
img_paths = sorted(img_paths, key=lambda x: float(x.split('/')[-1].split('_')[-1][:-4]))


npy_path = glob('datasets/UPfall/Subject4Activity1Trial2/squences/*_gt.npy')
npy_path = sorted(npy_path, key=lambda x: int(x.split('/')[-1].split('_')[0]))
for i, path in enumerate(npy_path):
    path = path.replace('_gt', '')
    img_path = img_paths[i]

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    data = np.load(path)[0]
    data[:, 0] *= w
    data[:, 1] *= h
    data: np.ndarray
    for pt in data.astype(int):
        img = cv2.circle(img, pt, 3, (255, 0, 0), -1)



    cv2.imwrite(f'vis/{i}.jpg', img)
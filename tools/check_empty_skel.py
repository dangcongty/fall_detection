from glob import glob

import numpy as np

missing_on_sequence = 0
total_sequences = 0
missing_on_total = 0
for npy_path in glob('datasets/ur_fall/raw/*/squences/*.npy'):
    if '_gt' in npy_path:
        continue
    skels = np.load(npy_path)
    flag = False
    for skel in skels:
        if [-1, -1] in skel:
            missing_on_sequence += 1
            flag = True
        total_sequences += len(skel)
        
    if flag:
        missing_on_total += 1

print(f'missing_on_sequence: {missing_on_sequence/total_sequences}')
print(f'missing_on_total: {missing_on_total/len(glob("datasets/ur_fall/raw/*/squences/*.npy"))}')
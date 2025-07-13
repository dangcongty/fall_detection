import os
import subprocess

import gdown

os.makedirs('datasets/UPfall', exist_ok=True)

# Open and read the HTML file line by line
with open("tools/upfall.html", "r") as file:
    lines = file.readlines()

c = 0
for k, line in enumerate(lines):
    if 'Camera1' in line and "Camera1_OF" not in line:
        for j in range(k-10, k):
            l = lines[j]
            if '<h5>' in l:
                video_name = l.strip()[4:-5]
                break
        if 'unavailable' in line:
            continue
        link = line.strip()[14:-18].split('=')[1].split(';')[0]
        if os.path.exists(f'/media/ssd220/ty/fall_detection_data/UPfall/{video_name}.zip'):
            continue
        subprocess.call(['gdown', link, '-O', f'/media/ssd220/ty/fall_detection_data/UPfall/{video_name}.zip'])
        print(video_name)
    else:
        continue


from glob import glob

flag = False
for zippath in glob(f'/media/ssd220/ty/fall_detection_data/UPfall/*.zip'):
    print(zippath)
    if 'Subject11Activity8Trial3' in zippath or flag:
        flag = True
        print(f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall/{os.path.basename(zippath)[:-4]}')
        os.makedirs(f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall/{os.path.basename(zippath)[:-4]}', exist_ok=True)
        subprocess.call(["unzip", '-qq', zippath, '-d', f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall/{os.path.basename(zippath)[:-4]}'])
'''
/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/fall_dataset/UPfall | image
datasets/UPfall | image
/media/ssd220/ty/fall_detection_data/UPfall/ | zip
'''

'''
ID	Description
1	Falling forward using hands
2	Falling forward using knees
3	Falling backwards
4	Falling sideward
5	Falling sitting in empty chair
6	Walking
7	Standing
8	Sitting
9	Picking up an object
10	Jumping
11	Laying
'''



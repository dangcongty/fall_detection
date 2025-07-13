from glob import glob

from ultralytics import YOLO

model = YOLO('weights/yolo11x-pose.pt')
res = model(source='changeview.jpg',
            imgsz = 1280,
            device = 'cuda:0',
            conf = 0.1,
            save = True)[0]
res.save('test.jpg')
print()

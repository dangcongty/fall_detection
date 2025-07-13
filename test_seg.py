from ultralytics import YOLO

model = YOLO('yolo11x-pose.pt')
model.predict(source='datasets/UPfall/Subject4Activity1Trial2/2018-07-10T13_27_48.376407.png', 
              save = True,
              project = 'runs',
              imgsz = 1920)
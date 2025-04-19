import json

from ultralytics import YOLO


class SkeletonExtractor:
    def __init__(self, model_path: str, img_size: int = 640):
        self.model = YOLO(model_path, task = 'pose')
        self.img_size = img_size
        
    def extract_skeleton(self, image_path: str, data: dict):
        result = self.model.predict(source=image_path, 
                                     imgsz=self.img_size, 
                                     device = 'cuda:1',
                                     verbose = False)[0]
        skeletons_norm = []
        skeletons = []
        if len(result):
            for pt_norm, pt, conf in zip(result.keypoints.xyn[0], result.keypoints.xy[0], result.keypoints.conf[0]):
                if conf < 0.5:
                    skeletons_norm.append([-1, -1])
                    skeletons.append([-1, -1])
                else:
                    skeletons_norm.append(pt_norm.cpu().numpy().tolist())
                    skeletons.append(pt.cpu().numpy().astype(int).tolist())
        else:
            for _ in range(17):
                skeletons_norm.append([-1, -1])
                skeletons.append([-1, -1])
        data['skeletons_norm'].append(skeletons_norm)
        data['skeletons'].append(skeletons)
        return skeletons, data
    

if __name__ == "__main__":
    import os
    from glob import glob

    from tqdm import tqdm

    dataset = 'urfall'
    model_path = 'weights/yolo11m-pose.pt'  # Replace with your model path

    extractor = SkeletonExtractor(model_path=model_path)
    if dataset == 'le2i':
        for video_path in tqdm(glob('datasets/le2i/data/*/*')):
            data = {
                    'video_path': video_path,
                    'skeletons_norm': [],
                    'skeletons': []
                }
            img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            for img_path in tqdm(img_paths):
                skeletons, data = extractor.extract_skeleton(img_path, data)
            with open(f'{video_path}/skeletons.json', 'w') as f:
                json.dump(data, f)

    elif dataset == 'urfall':
        for video_path in tqdm(glob('datasets/ur_fall/raw/*')):
            data = {
                    'video_path': video_path,
                    'skeletons_norm': [],
                    'skeletons': []
                }
            img_paths = sorted(glob(f'{video_path}/images/*.jpg'), key=lambda x: int(x.split('/')[-1].split('_')[0]))
            for img_path in tqdm(img_paths):
                skeletons, data = extractor.extract_skeleton(img_path, data)
            with open(f'{video_path}/skeletons.json', 'w') as f:
                json.dump(data, f)
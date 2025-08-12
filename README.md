## How to use
* 1. dataset: frame_extractor => tracking_kalman => make_sequence

## [UPDATE 13/07/2025]
* Thêm bộ classifier sau contrastive module
* Sửa lỗi metrics (0.5, 0.8, 0.9)


## [UPDATE 12/08/2025]
* Hoàn thành version 1: Contrastive Learning, Kalman Filter, YOLO-pose, TSNE



## Data Structure

```
workspace
│
├── datasets
│   ├── le2i
│   │   ├── data
│   │   │   ├── Coffee_room_01
│   │   │   │   ├── video (1)
│   │   │   │   │   ├── images/*.jpg       # Extracted image frames
│   │   │   │   │   ├── sequences/*.npy    # Numpy arrays of sequences
│   │   │   ├── ...
│   │   ├── raw_data                       # Unprocessed/raw files
│   │   │   ├── ...
│   └── ur_fall
│   │   ├── dataset/*.mp4 # eg. adl-01-cam0.mp4 ..
│   │   ├── raw
│   │   │   ├── adl-01-cam0.mp4
│   │   │   │   ├── images/*.jpg       # Extracted image frames
│   │   │   │   ├── sequences/*.npy    # Numpy arrays
```



git@github.com:dangcongty/fall_simulation_blender.git


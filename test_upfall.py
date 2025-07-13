import os
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import FallDataset
from models.st_gcn import Model
from utils.loss import Loss
from utils.metrics import IoU, Metrics


def test(model, val_loader, device):
    model.eval()
    running_loss = []
    running_trans_loss = []
    # running_iou = []
    stored_preds = None
    stored_gts = None
    metrics = Metrics(thresholds = [0.5, 0.8, 0.9])
    for i, (inputs, labels, paths) in tqdm(enumerate(val_loader), total=len(val_loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs.unsqueeze(-1))
        outputs = outputs.view(outputs.shape[0], outputs.shape[2])

        if stored_preds is not None:
            stored_preds = torch.concat([stored_preds, outputs], dim=0)
            stored_gts = torch.concat([stored_gts, labels], dim=0)
        else:
            stored_preds = outputs
            stored_gts = labels

        
    metrics.pred = stored_preds
    metrics.gt = stored_gts
    precisions, recalls, f1s, specs, acc, auc = metrics()

    return np.mean(running_loss), precisions[-1], recalls[-1], f1s[-1]


device = 'cuda:0'
model = Model(
    in_channels=2,
    num_class=1,
    graph_args=dict(layout='coco', strategy='spatial', max_hop=1),
    edge_importance_weighting=False,
    dropout=0.5).to(device)


model.load_state_dict(torch.load('/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Ty/checkpoints/ver57/model_epoch_7_recall.pth')['model'])
model.eval()
with open('datasets/UPfall/val.txt', 'r') as f:
    paths = f.readlines()

stored_preds = None
stored_gts = None

for k, label_path in enumerate(tqdm(paths)):
    label_path = label_path.strip()
    name = label_path.split('/')[-3]
    if k == 0:
        old_name = name
        
    if old_name != name:
        metrics = Metrics(thresholds = [0.5, 0.8, 0.9])
        metrics.pred = stored_preds
        metrics.gt = stored_gts
        precisions, recalls, f1s, specs, acc, auc = metrics()
        if recalls[0] == 0 and precisions[0] == 0:
            # no positive
            if (specs[0] + specs[1] + specs[2])/3 < 0.95:
                print('no positive', old_name)
        else:
            if (f1s[0] + f1s[1] + f1s[2])/3 < 0.9:
                print('positive', old_name)

        old_name = name
        stored_preds = None
        stored_gts = None

    skel_path = label_path.replace('_gt', '')

    skel = np.load(skel_path)
    label = np.load(label_path)

    label = torch.from_numpy(label).float().to(device).unsqueeze(0)
    skel = torch.from_numpy(skel).float().permute(2, 0, 1).to(device)  # T, V, C -> C, T, V

    with torch.no_grad():
        outputs = model(skel.unsqueeze(0).unsqueeze(-1))
    outputs = outputs.view(outputs.shape[0], outputs.shape[2])

    if stored_preds is not None:
        stored_preds = torch.concat([stored_preds, outputs], dim=0)
        stored_gts = torch.concat([stored_gts, label], dim=0)
    else:
        stored_preds = outputs
        stored_gts = label
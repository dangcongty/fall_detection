import numpy as np
import torch


def IoU(pred, gt):
    '''
    pred: B x T x 3
    gt: B x T x 3
    '''
    mask = gt[:, :, 0].to(torch.bool)  # B x T x 1
    
    pred = torch.clamp(pred, min=0)

    pred_length = pred[:, :, 1]
    gt_length = gt[:, :, 1]

    pred_center = pred[:, :, 2]
    gt_center = gt[:, :, 2]

    pred_start = pred_center - pred_length / 2
    pred_end = pred_center + pred_length / 2

    gt_start = gt_center - gt_length / 2
    gt_end = gt_center + gt_length / 2

    # Calculate the intersection
    intersection_start = torch.max(pred_start, gt_start)
    intersection_end = torch.min(pred_end, gt_end)
    intersection_length = torch.clamp(intersection_end - intersection_start, min=0)
    # Calculate the union   
    union_length = pred_length + gt_length - intersection_length
    torch.clamp(union_length, min=1e-3)
    iou = (intersection_length / (union_length+1e-6))
    if mask.sum() == 0:
        return [] # no object
    
    iou = iou[mask]
    return iou.cpu().numpy().tolist()

class Metrics(object):
    def __init__(self, thresholds = [0.5, 0.8, 0.9]):
        self.pred = []
        self.gt = []
        self.thresholds = thresholds

    def __call__(self):
        recalls = []
        precisions = []
        f1s = []
        gt = self.gt.bool()
        for th in self.thresholds:
            pred = self.pred > th
            self.TP = (pred & gt).sum().float()
            self.FP = (pred & ~gt).sum().float()
            self.FN = (~pred & gt).sum().float()
            
            recall = self.TP/(self.TP + self.FN + 1e-8)
            precision = self.TP/(self.TP + self.FP + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        return precisions, recalls, f1s

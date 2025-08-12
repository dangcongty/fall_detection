import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
    def __init__(self, threshold):
        self.cls = []
        self.ts = []
        self.gt = []
        self.threshold = threshold

    @staticmethod
    def find_sequences(arr, distance=3):
        sequences = []
        n = len(arr)
        
        start = None
        zero_count = 0

        for i, val in enumerate(arr):
            if val == 1:
                if start is None:
                    start = i
                zero_count = 0
            else:
                if start is not None:
                    zero_count += 1
                    if zero_count > distance:
                        sequences.append((start, i - zero_count))
                        start = None
                        zero_count = 0

        # Final check if a sequence continued till the end
        if start is not None:
            sequences.append((start, n - 1))

        return sequences

    @staticmethod
    def temporal_iou(gt_interval, pred_interval):
        start_i = max(gt_interval[0], pred_interval[0])
        end_i = min(gt_interval[1], pred_interval[1])
        intersection = max(0, end_i - start_i + 1)

        start_u = min(gt_interval[0], pred_interval[0])
        end_u = max(gt_interval[1], pred_interval[1])
        union = end_u - start_u + 1

        return intersection / union

    def match_events(self, gt_intervals, pred_intervals, iou_threshold=0.6):
        matched_gt = set()
        matched_pred = set()
        ious = []

        for i, pred in enumerate(pred_intervals):
            for j, gt in enumerate(gt_intervals):
                if j not in matched_gt:
                    iou = self.temporal_iou(gt, pred)
                    ious.append(iou)
                    if iou >= iou_threshold:
                        matched_gt.add(j)
                        matched_pred.add(i)
                        break

        TP = len(matched_pred)
        FP = len(pred_intervals) - TP
        FN = len(gt_intervals) - TP

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1, ious

    def __call__(self):
        pred_ts = torch.sigmoid(self.ts)
        pred_cls = torch.sigmoid(self.cls)


        pred_ts_binary = pred_ts>=0.5 # gaussian data 0.5 -> 1 -> 0.5
        pred_cls_binary = pred_cls>self.threshold


        pred = pred_ts_binary * pred_cls_binary

        gt_binary = self.gt >= 0.5 # gaussian data 0.5 -> 1 -> 0.5
        gt_intervals = [self.find_sequences((_gt).cpu().numpy()) for _gt in gt_binary]
        pred_intervals = [self.find_sequences((_pred).cpu().numpy()) for _pred in pred]

        results = {
            'idx': [],
            'p': [],
            'r': [],
            'f1': [],
            'iou': [],
        }
        for k, (_gt, _pred) in enumerate(zip(gt_intervals, pred_intervals)):
            if len(_gt) != 0:
                p, r, f1, iou = self.match_events(_gt, _pred, 0.5)
                results['idx'].append(k)
                results['p'].append(p)
                results['r'].append(r)
                results['f1'].append(f1)
                results['iou'].append(iou)

        precision = np.mean(results['p'])
        recall = np.mean(results['r'])
        f1 = np.mean(results['f1'])

        return precision, recall, f1

class MetricsV2(object):
    def __init__(self, threshold):
        self.pred = []
        self.gt = []
        self.threshold = threshold

    def __call__(self):
        pred_sigmoid = torch.sigmoid(self.pred)
        mse = ((pred_sigmoid - self.gt)**2).mean()
        return mse


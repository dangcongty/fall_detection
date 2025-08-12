import os

import cv2
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from models.st_gcn import Model
from train import Trainer
from utils.dataset_eval import FallDataset
from utils.metrics import IoU, Metrics


class Evaluation(Trainer):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = self.config['device']
        self.fixed_seed()
        self.init_model()
        self.init_loader()

        self.model.load_state_dict(torch.load(self.config['val']['ckpt'])['model'])
        self.vis_ct = False



    def init_model(self):
        fall_model_config = self.config['fall_model']
        graph_args_config = self.config['fall_model']['graph_args']
        self.use_contrastive_block = fall_model_config['use_contrastive_block']
        self.model = Model(
                in_channels=fall_model_config['in_channels'],
                num_class=fall_model_config['num_classes'],
                graph_args=dict(layout=graph_args_config['layout'], 
                                strategy=graph_args_config['strategy'], 
                                max_hop=graph_args_config['max_hop']),
                edge_importance_weighting=fall_model_config['edge_importance_weighting'],
                dropout=fall_model_config['dropout'], 
                use_ct=self.use_contrastive_block).to(self.device)

    def init_loader(self):
        self.data_config = self.config['data']
        val_file_name = 'test_kalman.txt' if self.data_config['use_kalman'] else 'test.txt'
        self.val_dataset = FallDataset(data_path=f'{self.data_config["root"]}/{val_file_name}')

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

    def extract_intervals(self, frame_preds):
        intervals = []
        start = None
        for i, val in enumerate(frame_preds):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                intervals.append((start, i - 1))
                start = None
        if start is not None:
            intervals.append((start, len(frame_preds) - 1))
        return intervals


    @staticmethod
    def find_sequences(arr, distance=12, length_fall=6):
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

        filter_sequences = []
        for sequence in sequences:
            s, e = sequence
            if e - s > length_fall:
                filter_sequences.append(sequence)

        # Final check if a sequence continued till the end
        if start is not None:
            filter_sequences.append((start, n - 1))

        return filter_sequences

    def __call__(self):
        self.tqdm_bar_format = '{l_bar}{bar:20}{r_bar}'
        self.model.eval()

        window_size = 32
        cross_valid = False
        threshold = 0.3  if not cross_valid else 0.1
        old_vid_id = -1
        frame_votes = []
        vote_counts = []
        frame_outs = []
        gts = []
        video_length = 0

        event_metrics = {}
        i = 0
        feat_layer1 = []
        feat_layer2 = []
        ct_labels = []
        fps = 25 if 'le2i' in self.data_config['root'] else 30
        self.threshold = 0.8
        for _, (inputs, labels, vid_id) in enumerate(tqdm(self.val_dataset(), bar_format=self.tqdm_bar_format)):
            # New video starts
            if vid_id != old_vid_id:
                if i != 0:
                    # Finalize previous video
                    frame_probs = [frame_votes[z] / vote_counts[z] if vote_counts[z] > 0 else 0 for z in range(video_length)]
                    frame_preds = np.array([1 if p >= threshold else 0 for p in frame_probs])
                    pred_intervals = self.find_sequences(frame_preds, length_fall=fps//3)

                    gt_intervals = self.find_sequences((gts>0)*1)

                    if len(gt_intervals) > 0:
                        for iou_th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                            precision, recall, f1, ious = self.match_events(gt_intervals, pred_intervals, iou_th)
                            if iou_th not in event_metrics:
                                event_metrics[iou_th] = []
                            event_metrics[iou_th].append((precision, recall, f1, ious))
                    
                    elif len(pred_intervals):
                        for iou_th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                            event_metrics[iou_th].append((0, 0, 0, 0))

                # Initialize new video
                process_length = len(self.val_dataset.reformat_paths[vid_id])
                video_length = len(self.val_dataset.reformat_paths[vid_id]) + 32
                frame_votes = np.array([0.0] * video_length)
                vote_counts = np.array([0] * video_length)
                frame_outs = []
                gts = np.array([0] * video_length)
                old_vid_id = vid_id
                i = 0

            # Inference
            inputs = inputs.to(self.device).unsqueeze(0)
            with torch.no_grad():
                if self.use_contrastive_block:
                    out_ts, out_cls, latents, ct_cls = self.model(inputs.unsqueeze(-1))

            out_cls = torch.sigmoid(out_cls).item()
            out_ts = torch.sigmoid(out_ts)

            cross = True

            self.threshold = 0.8 if not cross else 0.5
            pred_ts_binary = out_ts>=0.8 if not cross else out_ts>=0.5
            pred_cls_binary = out_cls>self.threshold
            pred = pred_ts_binary * pred_cls_binary

            frame_votes[i:i+32] += pred.cpu().numpy().flatten()
            vote_counts[i:i+32] += np.ones(32, dtype = np.int8)

            gt_binary = labels >= 0.5
            gts[i:i+32] += gt_binary


            feat_layer1.append(latents[0])
            feat_layer2.append(latents[1])
            ct_labels.append(labels.sum() > 0)

            i += 1

        # Final video
        if vote_counts.any():
            frame_probs = [frame_votes[z] / vote_counts[z] if vote_counts[z] > 0 else 0 for z in range(video_length)]
            frame_preds = np.array([1 if p >= threshold else 0 for p in frame_probs])
            pred_intervals = self.find_sequences(frame_preds)

            gt_intervals = self.find_sequences((gts>0)*1)

            if len(gt_intervals) > 0:
                for iou_th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    precision, recall, f1, ious = self.match_events(gt_intervals, pred_intervals, iou_th)
                    if iou_th not in event_metrics:
                        event_metrics[iou_th] = []
                    event_metrics[iou_th].append((precision, recall, f1, ious))

        # Average event-based metrics across all videos
        if event_metrics:
            for iou in event_metrics:

                all_precisions, all_recalls, all_f1s, ious = zip(*event_metrics[iou])
                mean_precision = np.mean(all_precisions)
                mean_recall = np.mean(all_recalls)
                mean_f1 = np.mean(all_f1s)

                print(f"Event-Based Evaluation at IoU: {iou}")
                print(f"  Precision: {mean_precision:.3f}")
                print(f"  Recall:    {mean_recall:.3f}")
                print(f"  F1 Score:  {mean_f1:.3f}")
        else:
            print("No predictions or ground truth found — check dataset or model.")

        if self.vis_ct or True:
            feat_layer1 = torch.concat(feat_layer1).squeeze().cpu().numpy()
            feat_layer2 = torch.concat(feat_layer2).squeeze().cpu().numpy()
            ct_labels = np.array(ct_labels)*1

            # Apply t-SNE
            tsne1 = TSNE(n_components=2, perplexity=50, random_state=self.config['seed'])
            tsne_result1 = tsne1.fit_transform(feat_layer1)

            tsne2 = TSNE(n_components=2, perplexity=50, random_state=self.config['seed'])
            tsne_result2 = tsne2.fit_transform(feat_layer2)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # First plot
            sns.scatterplot(
                x=tsne_result1[:, 0], y=tsne_result1[:, 1],
                hue=ct_labels, palette='tab10', s=60, ax=axes[0]
            )
            axes[0].set_title('t-SNE of Embeddings (Layer 1)')
            axes[0].legend(title='Class')

            # Second plot
            sns.scatterplot(
                x=tsne_result2[:, 0], y=tsne_result2[:, 1],
                hue=ct_labels, palette='tab10', s=60, ax=axes[1]
            )
            axes[1].set_title('t-SNE of Embeddings (Layer 2)')
            axes[1].legend(title='Class')

            plt.tight_layout()
            plt.savefig('urfall_urfall_ct.jpg')
            plt.close()

        
if __name__ == '__main__':
    evaluation = Evaluation('configs/base_config.yaml')
    evaluation()

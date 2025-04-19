import os

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


def validate(model: Model, val_loader, criterion, device):
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

        losses, mse_loss, trans_loss = criterion(outputs, labels)


        if stored_preds is not None:
            stored_preds = torch.concat([stored_preds, outputs], dim=0)
            stored_gts = torch.concat([stored_gts, labels], dim=0)
        else:
            stored_preds = outputs
            stored_gts = labels

        running_loss.append(losses.item())
        
    metrics.pred = stored_preds
    metrics.gt = stored_gts
    precision, recall, f1 = metrics()

    return np.mean(running_loss), precision, recall, f1

def main():

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    # if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)


    # Model, loss function, optimizer
    model = Model(
        in_channels=2,
        num_class=1,
        graph_args=dict(layout='coco', strategy='spatial', max_hop=1),
        edge_importance_weighting=False,
        dropout=0.5).to(device)
    
    model.load_state_dict(torch.load('checkpoints/le2i/model_epoch_891_recall.pth')['model']) # urfall

    criterion = Loss(device=device).to(device)

    # Data loaders
    val_dataset = FallDataset(data_path='datasets/ur_fall/all.txt')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


    # Validate the model (optional)
    val_loss, p, r, f1 = validate(model, val_loader, criterion, device)

    print(p)
    print(r)
    print(f1)


    print("Evaluating complete!")


if __name__ == "__main__":

    main()

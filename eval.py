import os

import cv2
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
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
    feats1 = []
    feats2 = []
    gts = []
    # running_iou = []
    stored_preds = None
    stored_gts = None
    metrics = Metrics(thresholds = [0.5, 0.8, 0.9])
    for i, (inputs, labels, paths, ct_label) in tqdm(enumerate(val_loader), total=len(val_loader)):
        inputs, labels, ct_label = inputs.to(device), labels.to(device), ct_label.to(device)

        with torch.no_grad():
            outputs, feat = model(inputs.unsqueeze(-1))
        outputs = outputs.view(outputs.shape[0], outputs.shape[2])

        losses, _ = criterion(outputs, labels, feat, ct_label)


        if stored_preds is not None:
            stored_preds = torch.concat([stored_preds, outputs], dim=0)
            stored_gts = torch.concat([stored_gts, labels], dim=0)
        else:
            stored_preds = outputs
            stored_gts = labels

        running_loss.append(losses.item())
        feats1.append(feat[0])
        feats2.append(feat[1])
        gts.append(ct_label)
        
    metrics.pred = stored_preds
    metrics.gt = stored_gts
    precision, recall, f1, specificity, acc, auc = metrics()

    return np.mean(running_loss), precision, recall, f1, (feats1, feats2, gts)

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
        edge_importance_weighting=True,
        dropout=0.5).to(device)
    
    # model.load_state_dict(torch.load('checkpoints/urfall_10072025/model_epoch_186_f1.pth')['model']) # urfall
    model.to(device)
    criterion = Loss(device=device).to(device)

    # Data loaders
    val_dataset = FallDataset(data_path='datasets/le2i/train.txt', transform=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    
    # Validate the model (optional)
    val_loss, p, r, f1, (feats1, feats2, gts) = validate(model, val_loader, criterion, device)

    print(p)
    print(r)
    print(f1)

    feats1 = torch.concat(feats1).squeeze().cpu().numpy()
    feats2 = torch.concat(feats2).squeeze().cpu().numpy()
    gts = torch.concat(gts).cpu().numpy()


    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_result = tsne.fit_transform(feats1)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=gts, palette='tab10', s=60)
    plt.title('t-SNE of Embeddings')
    plt.legend(title='Class')
    plt.savefig('tsne1.jpg')
    plt.clf()


    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_result = tsne.fit_transform(feats2)
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=gts, palette='tab10', s=60)
    plt.title('t-SNE of Embeddings')
    plt.legend(title='Class')
    plt.savefig('tsne2.jpg')
    plt.clf()


    print("Evaluating complete!")


    # load weight
    model.load_state_dict(torch.load('checkpoints/urfall_10072025/model_epoch_186_f1.pth')['model']) # urfall
    model.to(device)
    criterion = Loss(device=device).to(device)

    # Data loaders
    val_dataset = FallDataset(data_path='datasets/le2i/train.txt', transform=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    
    # Validate the model (optional)
    val_loss, p, r, f1, (feats1, feats2, gts) = validate(model, val_loader, criterion, device)

    print(p)
    print(r)
    print(f1)

    feats1 = torch.concat(feats1).squeeze().cpu().numpy()
    feats2 = torch.concat(feats2).squeeze().cpu().numpy()
    gts = torch.concat(gts).cpu().numpy()


    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_result = tsne.fit_transform(feats1)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=gts, palette='tab10', s=60)
    plt.title('t-SNE of Embeddings')
    plt.legend(title='Class')
    plt.savefig('tsne3.jpg')
    plt.clf()


    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_result = tsne.fit_transform(feats2)
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=gts, palette='tab10', s=60)
    plt.title('t-SNE of Embeddings')
    plt.legend(title='Class')
    plt.savefig('tsne4.jpg')
    plt.clf()


    print("Evaluating complete!")


    before_1 = cv2.imread('tsne1.jpg')
    before_2 = cv2.imread('tsne2.jpg')
    after_1 = cv2.imread('tsne3.jpg')
    after_2 = cv2.imread('tsne4.jpg')

    combine = np.vstack([np.hstack([before_1, after_1]), np.hstack([before_2, after_2])])
    cv2.imwrite('TSNE.jpg', combine)


if __name__ == "__main__":
    main()

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


def train(model: Model, train_loader, criterion, optimizer, device, writer: SummaryWriter, epoch):
    model.train()
    running_loss = []
    running_trans_loss = []
    running_mse_loss = []
    for i, (inputs, labels, paths) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(-1))
        outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        losses, mse_loss, trans_loss = criterion(outputs, labels)
        losses.backward()
        optimizer.step()

        running_loss.append(losses.item())
        running_trans_loss.append(trans_loss.item())
        running_mse_loss.append(mse_loss.item())

    writer.add_scalar('Train/Loss', np.mean(running_loss), global_step=epoch)
    writer.add_scalar('Train/Trans', np.mean(running_trans_loss), global_step=epoch)
    writer.add_scalar('Train/MSE', np.mean(running_mse_loss), global_step=epoch)
    return np.mean(running_loss)


def validate(model: Model, val_loader, criterion, device, writer: SummaryWriter, epoch):
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

    writer.add_scalar('Validation/Loss', np.mean(running_loss), epoch)

    for k, th in enumerate([0.5, 0.8, 0.9]):
        writer.add_scalar(f'Validation/precision_{th}', precision[k], epoch)
        writer.add_scalar(f'Validation/recall_{th}', recall[k], epoch)
        writer.add_scalar(f'Validation/f1_{th}', f1[k], epoch)

    return np.mean(running_loss), precision[-1], recall[-1], f1[-1]

def main():
    import shutil

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    # if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)

    # logger
    version = len(os.listdir('logs')) +1
    writer = SummaryWriter(log_dir=f'logs/ver{version}')
    os.makedirs(f'checkpoints/ver{version}')

    shutil.copy('train.py', f'checkpoints/ver{version}/train.py')
    shutil.copytree('models', f'checkpoints/ver{version}/models')
    shutil.copytree('data', f'checkpoints/ver{version}/data')
    shutil.copytree('utils', f'checkpoints/ver{version}/utils')

    # Model, loss function, optimizer
    model = Model(
        in_channels=2,
        num_class=1,
        graph_args=dict(layout='coco', strategy='spatial', max_hop=1),
        edge_importance_weighting=False,
        dropout=0.5).to(device)
    # model.load_state_dict(torch.load('checkpoints/ver32/model_epoch_891_recall.pth')['model'])

    criterion = Loss(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loaders
    train_dataset = FallDataset(data_path='datasets/ur_fall/train.txt')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)

    val_dataset = FallDataset(data_path='datasets/ur_fall/test.txt')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 1000
    best_p = 0
    best_r = 0
    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        # Validate the model (optional)
        val_loss, p, r, f1 = validate(model, val_loader, criterion, device, writer, epoch)

        if p > best_p:
            best_p = p
            torch.save({'model': model.state_dict(),
                        'precision': p,
                        'recall': r,
                        'f1': f1,
                        'loss': val_loss,
                        }, 
                        f'checkpoints/ver{version}/model_epoch_{epoch+1}_precision.pth')
            print(f"Model checkpoint saved at epoch {epoch+1}")

        if r > best_r:
            best_r = r
            torch.save({'model': model.state_dict(),
                        'precision': p,
                        'recall': r,
                        'f1': f1,
                        'loss': val_loss,
                        }, 
                        f'checkpoints/ver{version}/model_epoch_{epoch+1}_recall.pth')
            print(f"Model checkpoint saved at epoch {epoch+1}")
            
            
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model': model.state_dict(),
                        'precision': p,
                        'recall': r,
                        'f1': f1,
                        'loss': val_loss,
                        }, 
                        f'checkpoints/ver{version}/model_epoch_{epoch+1}_f1.pth')
            print(f"Model checkpoint saved at epoch {epoch+1}")


    print("Training complete!")


if __name__ == "__main__":

    main()

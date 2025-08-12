import os
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.st_gcn import Model
from utils.dataset import FallDataset
from utils.loss import Loss
from utils.metrics import IoU, Metrics
from utils.save_checkpoints import Saver


class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['device']
        self.fixed_seed()
        self.create_workspace_log()
        self.init_model()
        self.init_loss_optimizer()
        self.init_loader()
        self.init_ckpt_saver()

    def init_ckpt_saver(self):
        self.ckpt_saver = Saver(self.weight_dir)

    def init_loader(self):
        data_config = self.config['data']
        train_file_name = 'train.txt'
        val_file_name = 'test.txt'
        if data_config['use_kalman'] == True:
            train_file_name = 'train_kalman.txt'
            val_file_name = 'test_kalman.txt'

        train_dataset = FallDataset(data_path=f'{data_config["root"]}/{train_file_name}',
                                    arguments = data_config['augmentations'],
                                    mixup_data = data_config['mixup_data'],
                                    transform = True)
        self.train_loader = DataLoader(train_dataset, 
                                  batch_size=data_config['train_batch_size'], 
                                  pin_memory=True, 
                                  drop_last=True,
                                  sampler=train_dataset.sampler)

        val_dataset = FallDataset(data_path=f'{data_config["root"]}/{val_file_name}',
                                  arguments = data_config['augmentations'],
                                  mixup_data= data_config['mixup_data'],
                                  transform=False)
        self.val_loader = DataLoader(val_dataset, 
                                batch_size=data_config['val_batch_size'], 
                                shuffle=False)


    def init_loss_optimizer(self):
        loss_config = self.config['loss']
        train_config = self.config['train']
        self.criterion = Loss(device=self.device, 
                         use_focal_loss = loss_config['use_focal_loss'],
                         use_transition_loss = loss_config['use_transition_loss'],
                         use_contrastive_loss = loss_config['use_contrastive_loss'],
                         weight_loss = loss_config['weight_loss']
                         )
        optimizer = getattr(optim, train_config['optimizer']) 
        self.optimizer = optimizer(self.model.parameters(), lr=train_config['lr'])
        self.optimizer: optim.Optimizer
        
        self.num_epochs = train_config['num_epochs']

    def create_workspace_log(self):
        experiment_name = self.config['experiment_name']
        workspace_dir = self.config['workspace_dir']

        self.workdir = f'./{workspace_dir}/{experiment_name}'
        self.weight_dir = f'{self.workdir}/weights'

        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)
        shutil.copy('train.py', f'{self.workdir}/train.py')
        shutil.copytree('models', f'{self.workdir}/models', dirs_exist_ok = True)
        shutil.copytree('utils', f'{self.workdir}/utils', dirs_exist_ok = True)
        shutil.copytree('tools', f'{self.workdir}/tools', dirs_exist_ok = True)
        shutil.copytree('configs', f'{self.workdir}/configs', dirs_exist_ok = True)

        self.writer = SummaryWriter(log_dir=f'{self.workdir}/logs')

    def fixed_seed(self):
        seed = self.config['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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

    def create_running_losses(self):
        epoch_losses = {
            'classify_loss': [],
            'timestamp_loss': [],
            'transition_loss': [],
            'contrastive_loss': [],
            'contrastive_classify_loss': [],
            'total_loss': [],
        }
        return epoch_losses

    def train_one_epoch(self, epoch):
        epoch_train_losses = self.create_running_losses()
        self.model.train()
        for i, (inputs, labels, paths, ct_label) in enumerate(tqdm(self.train_loader, bar_format=self.tqdm_bar_format)):
            inputs, ct_label = inputs.to(self.device), ct_label.to(self.device)
            labels = [lb.to(self.device) for lb in labels]
            # forward
            self.optimizer.zero_grad()
            if self.use_contrastive_block:
                out_ts, out_cls, latents, ct_cls = self.model(inputs.unsqueeze(-1))
            else:
                outputs = self.model(inputs.unsqueeze(-1))

            # get loss
            out_ts: torch.Tensor
            out_ts = out_ts.view(out_ts.shape[0], out_ts.shape[2])
            losses = self.criterion(out_ts, out_cls, labels, latents, ct_label, ct_cls) if self.use_contrastive_block else self.criterion(outputs, labels, [], [], []) 
            
            # update
            losses: dict
            total_loss = losses.get('total_loss')
            total_loss: torch.Tensor
            total_loss.backward()
            self.optimizer.step()

            # stored
            for loss_name in losses:
                try:
                    epoch_train_losses[loss_name].append(losses[loss_name].item())
                except AttributeError as e:
                    continue
        # log
        for loss_name in losses:
            try:
                self.writer.add_scalar(f'Train/{loss_name}', np.mean(epoch_train_losses[loss_name]), global_step=epoch)
            except AttributeError as e:
                continue

    def validation(self, epoch):
        epoch_val_losses = self.create_running_losses()
        self.model.eval()
        metrics = Metrics(threshold = self.config['val']['threshold']['classify'])
        metric_cls = None
        metric_gts = None
        metric_ts = None
        for i, (inputs, labels, paths, ct_label) in enumerate(tqdm(self.val_loader, bar_format=self.tqdm_bar_format)):
            inputs, ct_label = inputs.to(self.device), ct_label.to(self.device)
            labels = [lb.to(self.device) for lb in labels]

            # forward
            with torch.no_grad():
                if self.use_contrastive_block:
                    out_ts, out_cls, latents, ct_cls = self.model(inputs.unsqueeze(-1))
                else:
                    raise "USE CONTRASTIVE BLOCK"
                    outputs = self.model(inputs.unsqueeze(-1))

            # get loss
            out_ts: torch.Tensor
            out_ts = out_ts.view(out_ts.shape[0], out_ts.shape[2])
            losses = self.criterion(out_ts, out_cls, labels, latents, ct_label, ct_cls) if self.use_contrastive_block else self.criterion(outputs, labels, [], [], []) 
            
            # stored result
            # raise "Metric error"
            if metric_cls is not None:
                metric_cls = torch.concat([metric_cls, out_cls], dim=0)
                metric_gts = torch.concat([metric_gts, labels[1]], dim=0)
                metric_ts = torch.concat([metric_ts, out_ts], dim=0)
            else:
                metric_cls = out_cls
                metric_gts = labels[1]
                metric_ts = out_ts

            # stored
            for loss_name in losses:
                try:
                    epoch_val_losses[loss_name].append(losses[loss_name].item())
                except AttributeError as e:
                    continue
        # log
        for loss_name in losses:
            try:
                self.writer.add_scalar(f'Val/{loss_name}', np.mean(epoch_val_losses[loss_name]), global_step=epoch)
            except AttributeError as e:
                continue

        metrics.ts = metric_ts
        metrics.cls = metric_cls
        metrics.gt = metric_gts
        precision, recall, f1 = metrics()
        self.writer.add_scalar(f'Validation/precision', precision, epoch)
        self.writer.add_scalar(f'Validation/recall', recall, epoch)
        self.writer.add_scalar(f'Validation/f1', f1, epoch)

        # save checkpoint
        self.ckpt_saver(self.model, epoch, precision, recall, f1, np.mean(epoch_val_losses['total_loss']))

    
    def train(self):
        self.tqdm_bar_format='{l_bar}{bar:20}{r_bar}'
        for epoch in range(self.num_epochs):
            print(f'Epoch: {epoch}')
            self.train_one_epoch(epoch)
            self.validation(epoch)


if __name__ == "__main__":
    trainer = Trainer('configs/base_config.yaml')
    trainer.train()

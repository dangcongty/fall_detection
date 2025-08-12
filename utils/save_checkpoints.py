

import torch

from models.st_gcn import Model


class Saver:
    def __init__(self, weight_dir):
        self.reset()
        self.weight_dir = weight_dir

    
    def reset(self):
        self.best_ckpts = {
            "recall": 0,
            "precision": 0,
            "f1": 0
        }

    def save_ckpt(self, model:Model, epoch, name, p, r, f1, val_loss):
        torch.save({'model': model.state_dict(),
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                    'loss': val_loss,
                    }, 
                    f'{self.weight_dir}/model_epoch_{epoch+1}_{name}.pth')

    def __call__(self, model, epoch, p, r, f1, val_loss):
        if p > self.best_ckpts['precision']:
            self.best_ckpts['precision'] = p
            self.save_ckpt(model, epoch, 'precision', p, r, f1, val_loss)
        if r > self.best_ckpts['recall']:
            self.best_ckpts['recall'] = r
            self.save_ckpt(model, epoch, 'recall', p, r, f1, val_loss)
        if f1 > self.best_ckpts['f1']:
            self.best_ckpts['f1'] = f1
            self.save_ckpt(model, epoch, 'f1', p, r, f1, val_loss)



class SaverV2:
    def __init__(self, weight_dir):
        self.reset()
        self.weight_dir = weight_dir

    
    def reset(self):
        self.best_ckpts = {
            "mse": 100
        }

    def save_ckpt(self, model:Model, epoch, name, mse, val_loss):
        torch.save({'model': model.state_dict(),
                    'mse': mse,
                    'loss': val_loss,
                    }, 
                    f'{self.weight_dir}/model_epoch_{epoch+1}_{name}.pth')

    def __call__(self, model, epoch, mse, val_loss):
        if mse < self.best_ckpts['mse']:
            self.best_ckpts['mse'] = mse
            self.save_ckpt(model, epoch, 'mse', mse, val_loss)
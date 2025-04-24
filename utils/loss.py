import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, device='cuda:1'):
        super(Loss, self).__init__()
        self.device = device

        self.MSE = nn.MSELoss()

    def transition_loss(self, output, target):
        out_diff = output[:, 1:] - output[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return self.MSE(out_diff, target_diff)
    
    def forward(self, output, target):
        '''
        output: B x T x 3
        target: B x T x 3
        '''
        loss = self.MSE(output, target)
        trans_loss = 0*self.transition_loss(output, target)
        total_loss = loss + trans_loss
        return total_loss, loss, trans_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda:1'):
        super(ContrastiveLoss, self).__init__()
        self.device = device

    def forward(self, output, target):
        '''
        output: B x T x 3
        target: B x T x 3
        '''
        if target.ndim == 2:
            target = target.squeeze(1)
        
        for b in range(target.shape[0]):
            out = output[b].unsqueeze(-1)
            out = F.normalize(out, dim=0)
            cosine_sim = out @ out.T

        label_eq = target.unsqueeze(0) == target.unsqueeze(1)


        return loss / (output.shape[0] * output.shape[1])

if __name__ == "__main__":
    out = torch.rand((2, 32)).to('cuda:1')
    target = torch.randint(0, 2, (2, 25)).to('cuda:1')

    # loss = Loss()
    loss = ContrastiveLoss()(out, target)
    loss(out, target)

    
import torch
import torch.nn as nn


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
        trans_loss = 10*self.transition_loss(output, target)
        total_loss = loss + trans_loss
        return total_loss, loss, trans_loss

if __name__ == "__main__":
    out = torch.rand((2, 25, 3)).to('cuda:1')
    target = torch.rand((2, 25, 3)).to('cuda:1')
    target[:, :, 0] = torch.randint(0, 2, (2, 25)).to('cuda:1')

    loss = Loss()
    loss(out, target)

    
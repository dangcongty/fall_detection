import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class Loss(nn.Module):
    def __init__(self, device='cuda:1', use_contrastive = False):
        super(Loss, self).__init__()
        self.device = device

        # self.MSE = nn.MSELoss(reduction = 'none')
        self.MSE = nn.MSELoss()
        self.CE = nn.BCELoss()
        self.focal_loss = FocalLoss()
        self.contrastive = ContrastiveLoss()
        self.use_contrastive = use_contrastive

    def transition_loss(self, output, target):
        out_diff = output[:, 1:] - output[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return self.MSE(out_diff, target_diff)
    
    def forward(self, output, target, latents, ct_label, out_cls):
        '''
        output: B x T x 3
        target: B x T x 3
        '''
        # alpha = 100
        # weights = torch.where(target.clone() == 1, alpha, 1)
        # loss = (self.MSE(output, target) * weights).sum()/weights.sum()
        loss = self.focal_loss(output, target)
        trans_loss = self.transition_loss(output, target)

        cons_loss = 0
        bce_loss = 0
        for k, (latent, out_c) in enumerate(zip(latents, out_cls)):
            cons_loss += self.contrastive(latent, ct_label)
            bce_loss += self.focal_loss(out_c.reshape((-1)), ct_label*1.0)

        # loss = torch.tensor([0]).to(self.device)
        # trans_loss = torch.tensor([0]).to(self.device)
        total_loss = loss + trans_loss + 10*cons_loss + bce_loss
        return total_loss, [loss, trans_loss, 10*cons_loss, bce_loss]


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, latent:torch.Tensor, labels):
        labels = labels.to(torch.float)
        # Compute pairwise cosine similarity
        normed = F.normalize(latent, p=2, dim=1)
        sim_matrix = torch.matmul(normed, normed.T)  # (B, B)
        
        # remove self pair
        mask = torch.eye(latent.size(0), dtype=torch.bool, device=latent.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)


        # Create positive mask: (i, j) = 1 if same class and i != j
        labels = labels.contiguous().view(-1, 1)
        positive_mask = (labels == labels.T) & ~mask  # [B, B]

        # For each anchor, compute log-softmax over similarities
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # Only keep positives
        mean_log_prob_pos = (log_prob * positive_mask).sum(1) / positive_mask.sum(1).clamp(min=1)

        # Final loss: mean over valid anchors
        loss = -mean_log_prob_pos.mean()

        return loss

    
if __name__ == "__main__":
    out = torch.rand((2, 32)).to('cuda:1')
    target = torch.randint(0, 2, (2, 25)).to('cuda:1')

    # loss = Loss()
    loss = ContrastiveLoss()(out, target)
    loss(out, target)

    
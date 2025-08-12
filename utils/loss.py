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
    def __init__(self, 
                 device='cuda:1', 
                 use_focal_loss = True,
                 use_transition_loss = False,
                 use_contrastive_loss = True,
                 weight_loss = [1, 1, 1],
                 ):
        super(Loss, self).__init__()
        self.device = device

        self.use_focal_loss = use_focal_loss
        self.use_transition_loss = use_transition_loss
        self.use_contrastive_loss = use_contrastive_loss
        self.weight_loss = weight_loss

        self.MSE = nn.MSELoss()
        self.CE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10, device=self.device))
        self.focal_loss = FocalLoss()
        self.contrastive = ContrastiveLoss()

    def transition_loss(self, output, target):
        out_diff = output[:, 1:] - output[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return self.MSE(out_diff, target_diff)
    
    def forward(self, out_ts, out_cls, labels, latents, ct_label, ct_cls):
        '''
        out_ts: B x T ==> final output timestamp
        out_cls: B x T ==> final output classify
        labels: 
            label_cls: B x T ==> label classify
            label_timestamp: B x T ==> soft-label timestamp
            soft_label_cls: B x 1 ==> soft-label classify

        latents: 2 x B x D ==> output latent feature from contrastive block
        ct_label: B ==> label classify
        ct_cls: 2 x B x 1 ==> output classify from contrastive block
        '''
        label_cls, label_timestamp = labels
        soft_label_cls = (label_cls.sum(1)/label_cls.shape[1]).unsqueeze(1)
        hard_label_cls = label_cls.max(1)[0].unsqueeze(1)
        classify_loss = self.CE(out_cls, hard_label_cls)

        out_ts = torch.sigmoid(out_ts)
        timestamp_loss = self.MSE(out_ts, label_timestamp)
        transition_loss = self.transition_loss(out_ts, labels) if self.use_transition_loss else 0
        
        if self.use_contrastive_loss:
            contrastive_loss = 0
            contrastive_classify_loss = 0
            for k, (latent, out_c) in enumerate(zip(latents[:-1], ct_cls)):
                contrastive_loss += self.contrastive(latent, ct_label)
                contrastive_classify_loss += self.CE(out_c, hard_label_cls)

            contrastive_loss += self.contrastive(latents[-1], ct_label)
            
            loss = {
                'timestamp_loss': self.weight_loss[0]*timestamp_loss,
                'classify_loss': self.weight_loss[1]*classify_loss,
                'transition_loss': self.weight_loss[2]*transition_loss,
                'contrastive_loss': self.weight_loss[3]*contrastive_loss,
                'contrastive_classify_loss': self.weight_loss[4]*contrastive_classify_loss,
                'total_loss': self.weight_loss[0]*timestamp_loss + \
                              self.weight_loss[1]*classify_loss + \
                              self.weight_loss[2]*transition_loss + \
                              self.weight_loss[3]*contrastive_loss + \
                              self.weight_loss[4]*contrastive_classify_loss
            }

        else:
            loss = {
                'timestamp_loss': self.weight_loss[0]*timestamp_loss,
                'classify_loss': self.weight_loss[1]*classify_loss,
                'transition_loss': transition_loss,
                'total_loss': self.weight_loss[0]*timestamp_loss + \
                              self.weight_loss[1]*classify_loss + \
                              self.weight_loss[2]*transition_loss
            }
        
        return loss


# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature = 0.07):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def _forward(self, latent:torch.Tensor, labels):
#         labels = labels.to(torch.float)
#         # Compute pairwise cosine similarity
#         normed = F.normalize(latent, p=2, dim=1)
#         sim_matrix = torch.matmul(normed, normed.T)  # (B, B)
        
#         # remove self pair
#         mask = torch.eye(latent.size(0), dtype=torch.bool, device=latent.device)
#         sim_matrix = sim_matrix.masked_fill(mask, -1e9)


#         # Create positive mask: (i, j) = 1 if same class and i != j
#         labels = labels.contiguous().view(-1, 1)
#         positive_mask = (labels == labels.T) & ~mask  # [B, B]

#         # For each anchor, compute log-softmax over similarities
#         log_prob = F.log_softmax(sim_matrix, dim=1)

#         # Only keep positives
#         mean_log_prob_pos = (log_prob * positive_mask).sum(1) / positive_mask.sum(1).clamp(min=1)

#         # Final loss: mean over valid anchors
#         loss = -mean_log_prob_pos.mean()

#         return loss

#     def forward(self, latent, labels):
#         # Normalize embeddings
#         normed = F.normalize(latent, p=2, dim=1)
#         sim_matrix = torch.matmul(normed, normed.T)

#         # mask self-comparisons
#         mask = torch.eye(latent.size(0), dtype=torch.bool, device=latent.device)

#         labels = labels.view(-1, 1)
#         positive_mask = (labels == labels.T) & ~mask  # positives only

#         # Compute exp(similarity)
#         exp_sim = torch.exp(sim_matrix) * ~mask  # remove self-pairs

#         # Denominator: sum over all except self
#         denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12

#         # Numerator: sum over positives
#         num = (exp_sim * positive_mask).sum(dim=1)

#         # Compute loss = -log(num/denom)
#         loss = -(torch.log(num + 1e-12) - torch.log(denom)).mean()

#         return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, feature, label):
        # feature: [batch, 512], label: [batch, 1] or [batch]
        batch_size = feature.size(0)
        feature = F.normalize(feature, p=2, dim=1)  # Normalize to unit vectors
        
        # Handle label dimensions
        if label.dim() == 2:
            label = label.squeeze(1)
        label = label.float()
        
        # Compute pairwise cosine similarities
        cosine_sim = torch.matmul(feature, feature.t())  # [batch, batch]
        
        # Create masks
        mask_diagonal = torch.eye(batch_size, device=feature.device).bool()
        mask_positive = (label.unsqueeze(1) == label.unsqueeze(0)) & ~mask_diagonal  # Same class, not diagonal
        mask_negative = (label.unsqueeze(1) != label.unsqueeze(0))  # Different class
        
        # Compute losses
        # Positive pairs: minimize distance (maximize similarity)
        # Loss = (1 - cosine_sim)^2 for similar pairs
        loss_positive = torch.pow(1 - cosine_sim, 2) * mask_positive.float()
        
        # Negative pairs: maximize distance (minimize similarity) with margin
        # Loss = max(0, cosine_sim + margin)^2 for dissimilar pairs
        loss_negative = torch.pow(torch.clamp(cosine_sim + self.margin, min=0.0), 2) * mask_negative.float()
        
        # Combine losses
        total_loss = loss_positive + loss_negative
        
        # Average over valid pairs
        num_positive_pairs = mask_positive.sum().float()
        num_negative_pairs = mask_negative.sum().float()
        total_pairs = num_positive_pairs + num_negative_pairs
        
        if total_pairs > 0:
            return total_loss.sum() / total_pairs
        else:
            return torch.tensor(0.0, device=feature.device, requires_grad=True)


def pairwise_cosine_contrastive_loss(latent, labels, margin=0.5):
    # Normalize latent vectors
    latent = F.normalize(latent, dim=1)
    
    # Compute pairwise cosine similarity: (64, 64)
    cosine_sim = torch.matmul(latent, latent.T)
    
    # Build pairwise label matrix: (64, 64)
    labels = labels.view(-1, 1)
    label_eq = (labels == labels.T).float()
    
    # Contrastive loss
    positive_loss = 1 - cosine_sim  # We want cosine_sim → 1 for same-label pairs
    negative_loss = F.relu(cosine_sim - margin)  # We want cosine_sim < margin for different-label pairs
    
    # Combine based on labels
    loss_matrix = label_eq * positive_loss + (1 - label_eq) * negative_loss

    # Exclude diagonal (self-comparisons)
    mask = ~torch.eye(latent.size(0), dtype=torch.bool, device=latent.device)
    loss = loss_matrix[mask].mean()

    return loss

class LossArch4(nn.Module):
    def __init__(self, 
                 device='cuda:1', 
                 use_focal_loss = True,
                 use_transition_loss = False,
                 use_contrastive_loss = True,
                 weight_loss = [1, 1, 1, 1],
                 weight_bce = 1,
                 ):
        super(LossArch4, self).__init__()
        self.device = device

        self.use_focal_loss = use_focal_loss
        self.use_transition_loss = use_transition_loss
        self.use_contrastive_loss = use_contrastive_loss
        self.weight_loss = weight_loss

        self.MSE = nn.MSELoss(reduction='none')
        self.CE = nn.BCEWithLogitsLoss()
        self.weight_CE = nn.BCEWithLogitsLoss(pos_weight=weight_bce)
        self.focal_loss = FocalLoss()
        self.contrastive = ContrastiveLoss()

        self.weight_bce = weight_bce

    def transition_loss(self, output, target):
        out_diff = output[:, 1:] - output[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return self.MSE(out_diff, target_diff)
    
    def forward(self, out_cls, labels, latents):
        '''
        out_ts: B x T ==> final output timestamp
        out_cls: B x T ==> final output classify
        labels: 
            label_cls: B x T ==> label classify
            label_timestamp: B x T ==> soft-label timestamp
            soft_label_cls: B x 1 ==> soft-label classify

        latents: 2 x B x D ==> output latent feature from contrastive block
        ct_label: B ==> label classify
        ct_cls: 2 x B x 1 ==> output classify from contrastive block
        '''
        label_cls, label_timestamp = labels

        out_cls = torch.sigmoid(out_cls)
        soft_label_cls = (label_cls.sum(1)/label_cls.shape[1]).unsqueeze(1)
        hard_label_cls = ((label_cls.sum(1)>0)*1.0).unsqueeze(1).flatten()
        weight = torch.ones_like(out_cls.flatten())+hard_label_cls*self.weight_bce
        classify_loss = weight*self.MSE(out_cls.flatten(), soft_label_cls.flatten())
        classify_loss = classify_loss.mean()

        ct_label = label_cls.sum(1) > 0
        if self.use_contrastive_loss:
            contrastive_loss = 0
            for k, latent in enumerate(latents):
                contrastive_loss += pairwise_cosine_contrastive_loss(latent, ct_label)

            loss = {
                'classify_loss': self.weight_loss[1]*classify_loss,
                'contrastive_loss': self.weight_loss[3]*contrastive_loss,
                'total_loss': 
                              self.weight_loss[1]*classify_loss + \
                              self.weight_loss[3]*contrastive_loss
            }

        return loss


if __name__ == "__main__":
    out = torch.rand((2, 32)).to('cuda:1')
    target = torch.randint(0, 2, (2, 25)).to('cuda:1')

    # loss = Loss()
    loss = ContrastiveLoss()(out, target)
    loss(out, target)

    
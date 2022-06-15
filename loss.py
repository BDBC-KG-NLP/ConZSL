import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Margin_Loss(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, margin=190., use_gpu=True):
        super(Margin_Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.use_gpu = use_gpu
        self.centers = nn.Embedding(num_classes, feat_dim)
        self.centers.weight.data.normal_(0.0, 0.02)
        if self.use_gpu:
            self.centers = self.centers.cuda()

    def forward(self, x, labels):
        center_scores = torch.sum(x*self.centers(labels), 1).unsqueeze(1)
        scores = torch.matmul(x, self.centers.weight.t())
        loss = torch.mean(torch.max(scores-center_scores+self.margin, torch.tensor(0.0).cuda()))
        return loss

class CrossEntropy_Loss(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CrossEntropy_Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        scores = torch.matmul(x, self.centers.t())
        loss = self.ce_loss(scores, labels)
        return loss

class SupContrastive_Loss(nn.Module):

    def __init__(self, tau=0.5):
        super(SupContrastive_Loss, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # Dot Product Kernel
        M = torch.matmul(x1, x2.t())/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x)==1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10) 
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss

class MI_Loss(nn.Module):

    def __init__(self, bound=0.5, alpha=1e-8):
        super(MI_Loss, self).__init__()
        self.bound = bound
        self.alpha = alpha

    def forward(self, mu, sigma):
        kl_divergence = (0.5 * torch.sum((mu ** 2) + (sigma ** 2)
                                      - torch.log((sigma ** 2) + self.alpha) - 1, dim=1))

        MI_loss = torch.max(torch.mean(kl_divergence) - self.bound, torch.tensor(0.).to(mu.device)) + self.bound

        return MI_loss
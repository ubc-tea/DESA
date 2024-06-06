import math
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isnan(log_prob)):
        #     log_prob[torch.isnan(log_prob)] = 0.0
        logging.info(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            # loss[torch.isnan(loss)] = 0.0
            logging.info(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
            raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class Distance_loss(nn.Module):
    def __init__(self, distance="SupCon", device=None):
        super(Distance_loss, self).__init__()
        self.distance = distance
        self.device = device
        if self.distance == "SupCon":
            self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)
        else:
            self.supcon_loss = None


    def forward(self, x1, x2, label1=None, label2=None):
        if self.distance == "L2_norm":
            loss = self.L2_norm(x1, x2)
        elif self.distance == "cosine":
            loss = self.cosine(x1, x2)
        elif self.distance == "SupCon":
            loss = self.supcon(x1, x2, label1, label2)
        else:
            raise NotImplementedError
        return loss


    def L2_norm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    def cosine(self, x1, x2):
        cos = F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos.mean()
        return loss

    def supcon(self, feature1, feature2, label1, label2):

        all_features = torch.cat([feature1, feature2], dim=0)

        all_features = F.normalize(all_features, dim=1)
        all_features = all_features.unsqueeze(1)

        align_cls_loss = self.supcon_loss(
            features=all_features,
            labels=torch.cat([label1, label2], dim=0),
            temperature=0.07, mask=None)
        return align_cls_loss











class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                        for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                    for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

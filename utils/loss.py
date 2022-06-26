import torch.nn.functional as F
from torch import nn
import torch


# class SoftDiceLoss(nn.Module):
#     def __init__(self):
#         super(SoftDiceLoss, self).__init__()
#
#     def forward(self, pred, targets):
#         num = targets.size(0)
#         smooth = 1
#
#         m1 = pred.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)
#
#         score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#         score = 1 - score.sum() / num
#         return score


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


class NpccLoss(nn.Module):
    def __init__(self, reduction=True):
        super(NpccLoss, self).__init__()
        self.reduce = reduction

    def forward(self, pred, target):
        target = target.view(target.size(0), target.size(1), -1)
        pred = pred.view(pred.size(0), pred.size(1), -1)

        vpred = pred - torch.mean(pred, dim=2).unsqueeze(-1)
        vtarget = target - torch.mean(target, dim=2).unsqueeze(-1)

        cost = - torch.sum(vpred * vtarget, dim=2) / \
               (torch.sqrt(torch.sum(vpred ** 2, dim=2))
                * torch.sqrt(torch.sum(vtarget ** 2, dim=2)))
        if self.reduce is True:
            return cost.mean()
        return cost


class RMSELoss(nn.Module):
    def __init__(self):
        self.reduce = 'mean'
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target, reduction=self.reduce))


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(), 1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label = label[mask]
        focal_weight = self.alpha * (label - pred).abs().pow(self.gamma) * (label > 0.0).float() + (
                    1 - self.alpha) * pred.abs().pow(self.gamma) * (label <= 0.0).float()
        loss = F.binary_cross_entropy(pred, label, reduction='none') * focal_weight
        return loss.sum() / pos_num



def cross_entropy(output, labels):
    loss = CrossEntropyLoss2d()
    return loss(output, labels)


def soft_dice(output, labels):
    loss = SoftDiceLoss()
    return loss(output, labels)


def npcc(output, labels):
    loss = NpccLoss()
    return loss(output, labels)


def rmse(output, labels):
    loss = RMSELoss()
    return loss(output, labels)


def bce(output, labels):
    loss = nn.BCEWithLogitsLoss()
    return loss(output, labels)

def fl(output, labels):
    loss = FocalLoss()
    return loss(output, labels)
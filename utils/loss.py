import torch.nn.functional as F
from torch import nn
import torch


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, targets):
        num = targets.size(0)
        smooth = 1

        m1 = pred.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


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


def cross_entropy(output, labels):
    loss = nn.CrossEntropyLoss(reduction='mean')
    return loss(output, torch.squeeze(labels).long())


def soft_dice(output, labels):
    loss = SoftDiceLoss()
    return loss(output, labels)


def npcc(output, labels):
    loss = NpccLoss()
    return loss(output, labels)


def rmse(output, labels):
    loss = RMSELoss()
    return loss(output, labels)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LABEL_SMOOTHING


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=LABEL_SMOOTHING):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        loss = -torch.sum(F.log_softmax(pred, 1) * smooth_one_hot, dim=1)
        return loss.mean()

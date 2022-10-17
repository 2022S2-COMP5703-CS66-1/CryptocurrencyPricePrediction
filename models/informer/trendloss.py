import torch
import torch.nn.functional as F


class TrendLoss(torch.nn.Module):

    def __init__(self, c=0.1, back_bone=F.mse_loss):
        super(TrendLoss, self).__init__()
        self.c = c
        self.back_bone = back_bone

    def forward(self, yhat, y):
        assert yhat.shape[-1] == 1
        back_bone_loss = self.back_bone(yhat, y)
        coef = 1 + self.c * torch.mean(torch.abs(yhat) / yhat - torch.abs(y) / y)
        return coef * back_bone_loss

import torch
import torch.nn.functional as F


class TrendLoss(torch.nn.Module):
    """
    Trend loss operator.
    It is using MSE as the base loss and apply extra punishment to the model
    when the model gives the prediction in the wrong trend.
    """
    def __init__(self, c=0.1, back_bone=F.mse_loss, eps=1e-8):
        super(TrendLoss, self).__init__()
        self.c = c
        self.back_bone = back_bone
        self.eps = eps

    def forward(self, yhat, y):
        assert yhat.shape[-1] == 1
        back_bone_loss = self.back_bone(yhat, y)
        coef = 1 + self.c * torch.mean(torch.abs(yhat) / (yhat + self.eps) - torch.abs(y) / (y + self.eps))
        return coef * back_bone_loss

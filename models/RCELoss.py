import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RCELoss(nn.Module):
    def __init__(self):
        super(RCELoss, self).__init__()
    
    # pred: (n, 2) one-hot matrix
    # target: (n, 2) model outputs
    def forward(self, target, pred):
        target = target.float()
        log_target = torch.log(target)
        log_target[log_target == float('-inf')] = -6

        loss = -torch.sum(pred*log_target) / pred.shape[0]
        return loss
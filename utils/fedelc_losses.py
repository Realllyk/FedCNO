import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super(LogitAdjust, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.tau = tau

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target)

def cross_entropy_soft(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def p_loss(pred, soft_targets_logits, target, alpha=0.1, beta=1.0):
    return pencil_loss(pred, soft_targets_logits, target, alpha=alpha, beta=beta)


def pencil_loss(pred, soft_targets_logits, target, alpha=0.5, beta=0.2):
    """
    PENCIL-style loss aligned with the FedELC baseline.
    pred: model output logits
    soft_targets_logits: learnable soft label logits (y_tilde)
    target: original noisy labels (indices)
    """
    pred_prob = F.softmax(pred, dim=1)
    log_soft_targets = F.log_softmax(soft_targets_logits, dim=1)

    # Lo: compatibility loss with original labels
    Lo = -torch.mean(log_soft_targets[torch.arange(soft_targets_logits.shape[0]), target])

    # Le: entropy loss on model predictions
    Le = -torch.mean(torch.sum(F.log_softmax(pred, dim=1) * pred_prob, dim=1))

    # Lc: consistency between soft labels and model predictions
    Lc = -torch.mean(torch.sum(log_soft_targets * pred_prob, dim=1)) - Le

    num_classes = soft_targets_logits.shape[1]
    loss_total = Lc / num_classes + alpha * Lo + beta * Le / num_classes
    return loss_total

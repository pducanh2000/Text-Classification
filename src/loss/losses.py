import torch
import torch.functional as F


def cross_entropy_with_soft_target(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets + F.log_softmax(pred), 1))


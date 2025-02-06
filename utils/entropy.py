import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import Optional


def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def Entropy(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy
    :param input: the softmax output
    :return: entropy
    """
    entropy = -input * torch.log(input + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def evidential_uncertainty(predictions, target, num_classes, device):
    predictions = predictions.to(device)
    target = target.to(device)

    # one hot encoding
    eye = torch.eye(num_classes).to(torch.float).to(device)
    labels = eye[target]
    # Calculate evidence
    evidence = F.softplus(predictions)

    # Dirichlet distribution paramter
    alpha = evidence + 1

    # Dirichlet strength
    strength = alpha.sum(dim=-1)

    # expected probability
    p = alpha / strength[:, None]

    # calculate error
    error = (labels - p) ** 2

    # calculate variance

    var = p * (1 - p) / (strength[:, None] + 1)

    # loss function
    loss = (error + var).sum(dim=-1)

    return loss.mean()


class TEntropyLoss(nn.Module):
    """
    The Tsallis Entropy for Uncertainty Reduction

    Parameters:
        - **t** Optional(float): the temperature factor used in TEntropyLoss
        - **order** Optional(float): the order of loss function
    """

    def __init__(self,
                 t: Optional[float] = 2.0,
                 order: Optional[float] = 2.0):
        super(TEntropyLoss, self).__init__()
        self.t = t
        self.order = order

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        entropy_weight = entropy_weight.repeat(1, n_class)
        tentropy = torch.pow(softmax_out, self.order) * entropy_weight
        # weight_softmax_out=softmax_out*entropy_weight
        tentropy = tentropy.sum(dim=0) / softmax_out.sum(dim=0)
        loss = -torch.sum(tentropy) / (n_class * (self.order - 1.0))
        return loss
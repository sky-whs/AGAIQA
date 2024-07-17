##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import pi, sqrt

safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))

class _BaseEntropyLoss2d(nn.Module):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
        """
        Parameters
        ----------
        ignore_index : Specifies a target value that is ignored
                       and does not contribute to the input gradient
        reduction : Specifies the reduction to apply to the output: 
                    'mean' | 'sum'. 'mean': elemenwise mean, 
                    'sum': class dim will be summed and batch dim will be averaged.
        use_weight : whether to use weights of classes.
        weight : Tensor, optional
                a manual rescaling weight given to each class.
                If given, has to be a Tensor of size "nclasses"
        """
        super(_BaseEntropyLoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_weights = use_weights
        if use_weights:
            print("w/ class balance")
            print(weight)
            self.weight = torch.FloatTensor(weight).cuda()
        else:
            print("w/o class balance")
            self.weight = None

    def get_entropy(self, pred, label):
        """
        Return
        ------
        entropy : shape [batch_size, h, w, c]
        Description
        -----------
        Information Entropy based loss need to get the entropy according to your implementation, 
        each element denotes the loss of a certain position and class.
        """
        raise NotImplementedError

    def forward(self, pred, label):
        """
        Parameters
        ----------
        pred: [batch_size, num_classes, h, w]
        label: [batch_size, h, w]
        """
        assert not label.requires_grad
        assert pred.dim() == 4
        assert label.dim() == 3
        assert pred.size(0) == label.size(0), "{0} vs {1} ".format(pred.size(0), label.size(0))
        assert pred.size(2) == label.size(1), "{0} vs {1} ".format(pred.size(2), label.size(1))
        assert pred.size(3) == label.size(2), "{0} vs {1} ".format(pred.size(3), label.size(3))

        n, c, h, w = pred.size()
        if self.use_weights:
            if self.weight is None:
                print('label size {}'.format(label.shape))
                freq = np.zeros(c)
                for k in range(c):
                    mask = (label[:, :, :] == k)
                    freq[k] = torch.sum(mask)
                    print('{}th frequency {}'.format(k, freq[k]))
                weight = freq / np.sum(freq) * c
                weight = np.median(weight) / weight
                self.weight = torch.FloatTensor(weight).cuda()
                print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = 1
        if self.ignore_index is None:
            self.ignore_index = c + 1

        entropy = self.get_entropy(pred, label)

        mask = label != self.ignore_index
        weighted_entropy = entropy * self.weight

        if self.reduction == 'sum':
            loss = torch.sum(weighted_entropy, -1)[mask].mean()
        elif self.reduction == 'mean':
            loss = torch.mean(weighted_entropy, -1)[mask].mean()
        return loss


class OrdinalRegression2d(_BaseEntropyLoss2d):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
        super(OrdinalRegression2d, self).__init__(ignore_index, reduction, use_weights, weight)

    def get_entropy(self, pred, label):
        n, c, h, w = pred.size()
        label = label.unsqueeze(3).long()
        pred = pred.permute(0, 2, 3, 1)
        mask10 = ((torch.arange(c)).cuda() <  label).float()
        mask01 = ((torch.arange(c)).cuda() >= label).float()
        entropy = safe_log(pred) * mask10 + safe_log(1 - pred) * mask01
        return -entropy

def NormalDist(x, sigma):
    f = torch.exp(-x**2/(2*sigma**2)) / sqrt(2*pi*sigma**2)
    return f


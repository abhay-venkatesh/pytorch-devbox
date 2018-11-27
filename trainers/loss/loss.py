import warnings

import torch
from .module import Module
from .container import Sequential
from .activation import LogSoftmax
from .. import functional as F
from ..functional import _Reduction


class _Loss(Module):
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='elementwise_mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class MSELoss(_Loss):
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction='elementwise_mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)
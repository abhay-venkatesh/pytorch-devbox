import numpy as np
import torch


def UnbiasedPULoss(X, A, rho=0.7):
    """ X: outputs
        A: labels
        rho: noise rate """
    X_ = (X - 1).pow(2)
    numer = X_ - (rho * (X.pow(2)))
    frac = (numer / (1 - rho))
    positive_case = frac * A
    zeroth_case = (1-A) * (X.pow(2))
    loss = positive_case + zeroth_case
    return loss.sum()


# Outputs from Model
X1 = torch.FloatTensor([[1, 1, 1], [0, 0, 0]])
X2 = torch.FloatTensor([[1, 1, 1], [0, 0, 0]])
X = torch.stack([X1, X2])

A1 = torch.FloatTensor([[0, 0, 0], [1, 1, 1]])
A2 = torch.FloatTensor([[0, 0, 0], [1, 1, 1]])
A = torch.stack([A1, A2])

loss = UnbiasedPULoss(X, A)
print(loss)

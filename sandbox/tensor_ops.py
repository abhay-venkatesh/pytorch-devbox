import numpy as np
import torch


def UnbiasedPULoss(X, A, rho=0.7):
    """ X: outputs
        A: labels
        rho: noise rate """
    width = X.size()[2]
    height = X.size()[1]
    return 0


# Outputs from Model
X_ = np.row_stack(([1, 0, 0], [1, 1, 1]))
X1 = torch.from_numpy(X_)
X2 = torch.from_numpy(X_)
X = torch.stack([X1, X2])

A_ = np.row_stack(([1, 0, 0], [1, 0, 0], [1, 0, 0]))
A1 = torch.from_numpy(A_)
A2 = torch.from_numpy(A_)
A = torch.stack([A1, A2])

loss = UnbiasedPULoss(X, A)
# print(loss)

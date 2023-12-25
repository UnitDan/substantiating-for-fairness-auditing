import torch
import numpy as np

def accuracy(model, test_dl, device='cpu'):

    model.eval()
    corr, total = 0, 0

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=1)
        total += y.shape[0]
        corr += torch.sum(y_pred == y)

    score = corr / float(total)
    return score
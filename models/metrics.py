import torch
import numpy as np

def accuracy(model, data_loader, device='cpu'):
    model.eval()
    model.to(device)
    corr, total = 0, 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        _, y_pred = torch.max(y_pred, dim=1)
        total += y.shape[0]
        corr += torch.sum(y_pred == y)

    score = corr / float(total)
    return score

def consistancy(model, X, X_counter, device='cpu'):
    model.eval()
    model.to(device)
    X = X.to(device)
    X_counter = X_counter.to(device)
    
    pred = np.array(model.get_prediction(X).cpu())
    pred_conter = np.array(model.get_prediction(X_counter).cpu())
    return np.mean(pred == pred_conter)

def accuracy_variance(model, X, Y, group, device='cpu'):
    model.eval()
    model.to(device)

    gs = group.unique()
    acc = []
    for g in gs:
        X_g = X[group == g].to(device)
        Y_g = Y[group == g].to(device)
        pred_g = model.get_prediction(X_g)
        acc.append(np.mean((pred_g == Y_g).cpu().numpy()))
    return max(acc) - min(acc)
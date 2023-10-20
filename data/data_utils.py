from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class ProtectedDataset(Dataset):
    def __init__(self, data, labels, protected_idxs):
        self.data = data
        self.labels = labels
        self.use_protected_attr = False
        self.protected_idxs = protected_idxs
        self.columns_to_keep = [i for i in range(data.shape[1]) if i not in protected_idxs]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if not self.use_protected_attr:
            data = self.data[index][self.columns_to_keep]
        else:
            data = self.data[index]
        label = self.labels[index]
        return data, label

    def dim_feature(self):
        if self.use_protected_attr:
            dim = self.data.shape[1]
        else:
            dim = len(self.columns_to_keep)
        return dim
    
    

def convert_df_to_tensor(data_X_df, data_Y_df):
    xx = data_X_df.values.astype(float)
    data_X = torch.tensor(xx).float()
    data_Y = torch.tensor(data_Y_df.values)

    return data_X, data_Y


def onehot_to_idx(x):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return torch.argmax(x, dim=1).unsqueeze(0).T

def idx_to_onehot(idx, n_choice):
    if len(idx.shape) > 1:
        idx = idx.squeeze(0)
    rows = idx.shape[0]
    x = torch.zeros((rows, n_choice))
    i = tuple(range(rows))
    j = tuple(idx.tolist())
    x[[i, j]] = 1
    return x

class Data_gen():
    def __init__(self) -> None: pass
    def gen_by_range(self, n=1): pass
    def gen_by_distribution(self, n=1): pass
    def clip(self, x): pass
    def random_perturb(self, x, dx=None, epsilon=None): pass
    def data_around(self, x_sample): pass
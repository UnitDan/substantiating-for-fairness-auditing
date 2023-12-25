from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from utils import UnfairMetric
import pandas as pd

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

    def get_all_data(self):
        if not self.use_protected_attr:
            data = self.data[:, self.columns_to_keep]
        else:
            data = self.data
        return data

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

class DataGenerator():
    def __init__(self, include_protected_feature):
        self.include_protected_feature = include_protected_feature

    def _initialize(self, sensitive_columns):
        self.sensitive_columns = sensitive_columns
        self.columns_to_keep = [i for i in range(self.X.shape[1]) if i not in self.sensitive_columns]
        self.data_range = torch.quantile(self.X, torch.Tensor([0, 1]), dim=0)
        
        self.all_features = self._data2feature(self.X)
        self.feature_range = torch.quantile(self.all_features, torch.Tensor([0, 1]), dim=0)

    def get_range(self, data_or_feature, include_protected_feature=None):
        if include_protected_feature == None:
            include_protected_feature = self.include_protected_feature
        if data_or_feature == 'data':
            if not include_protected_feature:
                return self.data_range[:, self.columns_to_keep]
            else:
                return self.data_range
        elif data_or_feature =='feature':
            return self.feature_range

    def gen_by_range(self, n=1):
        r = self.get_range('feature')
        l, u = r[0], r[1]
        features = torch.rand((n, self.all_features.shape[1]))
        features = torch.floor(l + features*(u - l + 1))
        # features = torch.floor(l + features*(u - l))
        data = self._feature2data(features)

        if not self.include_protected_feature:
            data = data[:, self.columns_to_keep]

        return data
    
    def gen_by_distribution(self, n=1):
        idxs = torch.randint(self.all_features.shape[0], (n, self.all_features.shape[1]))
        data = []
        for i in range(n):
            x = self.all_features[idxs[i], torch.arange(self.all_features.shape[1])]
            data.append(x)
        data = torch.concat(data, dim=0)
        data = self._feature2data(data)

        if not self.include_protected_feature:
            data = data[:, self.columns_to_keep]

        return data

    def clip(self, data, with_protected_feature=None):
        if with_protected_feature is None:
            with_protected_feature = self.include_protected_feature
        def _onehot(data):
            o = torch.zeros_like(data)
            o[torch.arange(data.shape[0]), torch.argmax(data, dim=1)] = 1
            return o
        
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        
        if not with_protected_feature:
            x = torch.zeros((data.shape[0], data.shape[1] + len(self.sensitive_columns)))
            x[:, self.columns_to_keep] = data
            data = x
        
        for r in self.onehot_ranges:
            data[:, r[0]: r[1]] = _onehot(data[:, r[0]: r[1]])
        data_range = self.get_range('data')
        continuous_low, continuous_high = data_range[0][self.continuous_columns], data_range[1][self.continuous_columns]
        data[:, self.continuous_columns] = data[:, self.continuous_columns].clip(continuous_low, continuous_high)
        # 不取整，其他一致

        if not with_protected_feature:
            data = data[:, self.columns_to_keep]

        return data
             
    def norm(self, x_sample):
        data_range = self.get_range('data')
        l, u = data_range[0], data_range[1]
        return (x_sample - l) / (u - l)
    
    def recover(self, x_sample):
        data_range = self.get_range('data')
        l, u = data_range[0], data_range[1]
        return (u - l)*x_sample + l
    
    def feature_dataframe(self, feature=None, data=None):
        if feature == None and data != None:
            feature = self._data2feature(data)
        elif data == None and feature == None:
            raise Exception('Parameters `feature` and `data` cannot both be `None`.')
        elif data != None and feature != None:
            raise Exception('The value of parameters `feature` and `data` cannot be specified simultaneously.')
        
        features = pd.DataFrame(feature.detach(), columns=self.feature_name, dtype='int64')
        return features

    def _data2feature(self, data):
        if len(data.shape) == 1:
            data = data.unsqueeze(dim=0)

        continous_feature = data[:, self.continuous_columns]
        onhot_features = []
        for i in range(len(self.onehot_ranges)):
            onhot_features.append(data[:, self.onehot_ranges[i][0]:self.onehot_ranges[i][1]])

        features = [continous_feature]
        for x in onhot_features:
            x = onehot_to_idx(x)
            features.append(x)
        features = torch.concat(features, dim=1)
        return features
    
    def _feature2data(self, feature):
        if len(feature.shape) == 1:
            feature = feature.unsqueeze(dim=0)

        data_length = self.X.shape[1]
        data = torch.zeros((feature.shape[0], data_length))
        
        continous_id = list(range(len(self.continuous_columns)))
        data[:, self.continuous_columns] = feature[:, continous_id]

        for i in range(len(self.onehot_ranges)):
            index = continous_id[-1] + 1 + i
            r = self.onehot_ranges[i]
            data[:, r[0]:r[1]] = idx_to_onehot(feature[:, index], r[1] - r[0])
        return data
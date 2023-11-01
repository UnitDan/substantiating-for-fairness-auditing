from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABCMeta, abstractmethod
import numpy as np
import random
from utils import Unfair_metric

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

class Data_gen(metaclass=ABCMeta):
    def __init__(self, include_protected_feature):
        self.include_protected_feature = include_protected_feature

    def _initialize(self, sensitive_columns):
        self.sensitive_columns = sensitive_columns
        self.columns_to_keep = [i for i in range(self.X.shape[1]) if i not in self.sensitive_columns]
        self.data_range = torch.quantile(self.X, torch.Tensor([0, 1]), dim=0)
        
        self.all_features = self._data2feature(self.X)
        self.feature_range = torch.quantile(self.all_features, torch.Tensor([0, 1]), dim=0)

    @abstractmethod
    def _data2feature(self, x): pass

    @abstractmethod
    def _feature2data(self, x): pass
    
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
        features = (l + features*(u - l)).to(torch.int)
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
        data = torch.round(data)

        if not with_protected_feature:
            data = data[:, self.columns_to_keep]

        return data
    
    def all_possible_perturb_step(self, data_sample, unfair_metric:Unfair_metric):
        data_sample = data_sample.squeeze()
        
        if not self.include_protected_feature:
            x = torch.zeros((data_sample.shape[0] + len(self.sensitive_columns)))
            x[self.columns_to_keep] = data_sample
            data_sample = x

        feature_origin = self._data2feature(data_sample).squeeze()
        pert = torch.eye(feature_origin.shape[0])
        pert = torch.concat([pert, -pert], dim=0)
        pert_feature = feature_origin + pert
        
        feature_range = self.get_range('feature')
        range_filter = torch.logical_and(pert_feature>=feature_range[0], pert_feature<=feature_range[1])
        pert_feature = pert_feature[torch.all(range_filter, dim=1)]

        pert_data = self._feature2data(pert_feature)
        if not self.include_protected_feature:
            pert_data = pert_data[:, self.columns_to_keep]
            data_sample = data_sample[self.columns_to_keep]
            # filter out the perturbation on sensitive features
            pert_data = pert_data[torch.logical_not(torch.all(data_sample, pert_data), dim=1)]

        distances = unfair_metric.dx(data_sample.unsqueeze(0), pert_data, itemwise_dist=False).squeeze()
        # print(len(pert_data), distances)
        pert_data = pert_data[distances <= 1/unfair_metric.epsilon]
        # print(len(pert_data))

        

        return pert_data

    def random_perturb(self, data_sample, unfair_metric:Unfair_metric):
        searched = torch.Tensor()
        while 1:
            step_pert = self.all_possible_perturb_step(data_sample, unfair_metric)

            set1 = {tuple(row.numpy()) for row in searched}
            set2 = {tuple(row.numpy()) for row in step_pert}
            if len(set1) == 0 and len(set2) == 0:
                raise Exception(f'data sample has no possible perturbation with in dx <= 1/epsilon ({1/unfair_metric.epsilon})')
            
            new_pert = torch.Tensor(list(set2 - set1))
            if new_pert.shape[0] == 0:
                break

            pert_data = step_pert[random.randint(0, step_pert.shape[0]-1)]
            searched = torch.concat([searched, pert_data.unsqueeze(0)], dim=0)

            if random.random() > 0.5:
                break

        return pert_data
 
    def data_around(self, x_sample):
        x_sample = x_sample.squeeze()

        ceil = torch.ceil(x_sample[self.continuous_columns])
        floor = torch.floor(x_sample[self.continuous_columns])
        cont_matrix = torch.concat([ceil, floor]).view(2, -1)
        choices = [torch.unique(cont_matrix[:, i]) for i in range(cont_matrix.shape[1])]
        for r in self.onehot_ranges:
            x_onehot = x_sample[r[0]: r[1]]
            choices.append(torch.where(x_onehot)[0].float())
        features = torch.cartesian_prod(*choices)
        datas = self._feature2data(features)
        return datas
import torch
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm.auto import tqdm
import random
from torch.utils.data import SubsetRandomSampler
from data.data_utils import ProtectedDataset

def load_model(model, dataset_name, trainer_name, use_sensitive_attr, sensitive_vars, id, note='', path='new'):
    root_dir = os.path.join('trained_models', path)
    file_name = f'MLP_{dataset_name}_{trainer_name}_{"all-features" if use_sensitive_attr else "without-"+"-".join(sensitive_vars)}_{id}{note}'
    model.load(os.path.join(root_dir, file_name))

def get_data(data, rand_seed, sensitive_vars):
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)

    X, y, sensitive_idxs = data.load_data(sensitive_vars=sensitive_vars)

    # randomly split into train/test splits
    total_samples = len(X)
    train_size = int(total_samples * 0.8)

    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    dataset = ProtectedDataset(X, y, sensitive_idxs)
    train_loader = DataLoader(dataset, batch_size=512, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1000, sampler=test_sampler)

    return dataset, train_loader, test_loader

class UnfairMetric():
    def __init__(self, dx, dy, epsilon) -> None:
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
    
    def is_unfair(self, x1, x2, y1, y2):
        return (self.dy(y1, y2).item() > self.dx(x1, x2).item()*self.epsilon)
    
def get_L_matrix(all_X, all_pred, dx, dy):
    ds = TensorDataset(all_X, all_pred)
    dl = DataLoader(ds, batch_size=3000, shuffle=False)
    L = []
    for b in tqdm(dl):
        dxs = dx(b[0], all_X, itemwise_dist=False)
        dys = dy(b[1], all_pred, itemwise_dist=False)

        L_batch = (dys/dxs).squeeze()
        L.append(L_batch)
    L = torch.concat(L, dim=0)
    return L
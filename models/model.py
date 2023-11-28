import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fcout = nn.Linear(100, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcout(x)
        return x
    
    def get_prediction(self, batch):
        self.eval()
        if len(batch.shape) == 1: # one sample is passed to param:batch
            batch = batch.unsqueeze(0) # 1 * D

        y_pred = self.forward(batch)
        _, y_pred = torch.max(y_pred, dim=1)
        return y_pred

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        device = torch.device('cpu')
        self.load_state_dict(torch.load(load_path, map_location=device))

class MaxMinNormalizationLayer(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = torch.nn.Parameter(min_value, requires_grad=False)
        self.max_value = torch.nn.Parameter(max_value, requires_grad=False)

    def forward(self, x):
        # 执行Max-Min归一化操作
        normalized_data = (x - self.min_value) / (self.max_value - self.min_value)
        return normalized_data

class NormMLP(MLP):
    def __init__(self, input_size, output_size, data_range):
        super().__init__(input_size, output_size)
        self.norm = MaxMinNormalizationLayer(data_range[0], data_range[1])

    def forward(self, x):
        x = self.norm(x)
        return super().forward(x)

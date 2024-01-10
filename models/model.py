import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size, data_gen, n_layers=2, norm=False):
        super().__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, 64))
        for _ in range(n_layers - 1):
            self.fc.append(nn.Linear(64, 64))
        self.fcout = nn.Linear(64, output_size)
        self.data_gen = data_gen
        self.norm = norm

    def forward(self, x):
        if self.norm:
            x = self.data_gen.norm(x)
        for layer in self.fc:
            x = F.relu(layer(x))
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
        if save_path.split('.')[-1] != 'pth':
            save_path += '.pth'
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        if load_path.split('.')[-1] != 'pth':
            load_path += '.pth'
        device = torch.device('cpu')
        self.load_state_dict(torch.load(load_path, map_location=device))
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
import pickle

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 100)
        self.fcout = nn.Linear(100, output_size)
        self.to('cpu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
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

class RandomForest(nn.Module):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.random_forest = RandomForestClassifier(n_estimators=n_estimators, max_features=10, max_depth=max_depth)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.cpu().detach().numpy()
        return torch.from_numpy(self.random_forest.predict_proba(x))
    
    def get_prediction(self, batch):
        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)
        batch = batch.cpu().detach().numpy()
        return torch.from_numpy(self.random_forest.predict(batch)).int()

    def save(self, save_path):
        if save_path.split('.')[-1] != 'pkl':
            save_path += '.pkl'
        with open(save_path, 'wb') as file:
            pickle.dump(self.random_forest, file)

    def load(self, load_path):
        if load_path.split('.')[-1] != 'pkl':
            load_path += '.pkl'
        with open(load_path, 'rb') as file:
            self.random_forest = pickle.load(file)
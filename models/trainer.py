import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.metrics import accuracy
from models.model import MLP, RandomForest

class STDTrainer():
    def __init__(self, model, train_dl, test_dl, device, epochs, lr):
        self.model = model
        if not isinstance(model, MLP):
            raise Exception('Expect the model to be a MLP.')
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = F.cross_entropy
    
    def train(self):
        self.model.to(self.device)

        self.model.train()
        for e in tqdm(range(self.epochs)):
            for x, y in self.train_dl:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x).squeeze()
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
        
        ac = accuracy(self.model, self.train_dl, self.device)
        print(f'Train Accuracy: {ac}')

        ac = accuracy(self.model, self.test_dl, self.device)
        print(f'Test Accuracy: {ac}')

class RandomForestTrainer():
    def __init__(self, model, train_dl, test_dl):
        self.model = model
        if not isinstance(model, RandomForest):
            raise Exception('Expect the model to be a RandomForest.')
        self.train_dl = train_dl
        self.test_dl = test_dl

    def train(self):
        X_train, y_train = torch.Tensor(), torch.Tensor()
        for X, y in self.train_dl:
            X_train = torch.concat([X_train, X])
            y_train = torch.concat([y_train, y])
        
        X_train = X_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        self.model.random_forest.fit(X_train, y_train)

        ac = accuracy(self.model, self.train_dl)
        print(f'Train Accuracy: {ac}')

        ac = accuracy(self.model, self.test_dl)
        print(f'Test Accuracy: {ac}')
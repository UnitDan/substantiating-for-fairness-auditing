import torch
import torch.nn.functional as F
from tqdm import tqdm
from dnn_models.metrics import accuracy

class STDTrainer():
    def __init__(self, model, train_dl, test_dl, device, epochs, lr):
        self.model = model
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
        
        ac = accuracy(self.model, self.test_dl, self.device)
        print(f'Accuracy: {ac}')


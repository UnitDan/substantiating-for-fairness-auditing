import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from models.metrics import accuracy
from inFairness.auditor import SenSeIAuditor

class STDTrainer():
    def __init__(self, model, train_dl, test_dl, device, epochs, lr):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = F.cross_entropy
    
    def train(self, patience=20):
        self.model.to(self.device)

        best_ac = None
        counter = 0
        # for e in tqdm(range(self.epochs)):
        for e in range(self.epochs):
            self.model.train()
            for x, y in self.train_dl:
                # print(x, y, sep='\n')
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x).squeeze()
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
            ac = accuracy(self.model, self.test_dl, self.device)
            if best_ac == None or ac - best_ac > 1e-5:
                best_ac = ac
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break
        print('epoches:', e, '\n')

class SenSeiTrainer():
    '''
    rho越大, 越倾向于公平性优化而牺牲性能。
    rho=0则退化为普通的优化器。
    '''
    def __init__(self, model, train_dl, test_dl, device, epochs, lr, distance_x, distance_y, rho=5.0, eps=0.1, auditor_nsteps=50, auditor_lr=1e-3):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = F.cross_entropy

        self.lamb = None
        self.rho = rho
        self.eps = eps
        self.distance_x = distance_x
        self.distance_y = distance_y
        self.distance_x.to(self.device)
        self.distance_y.to(self.device)
        
        self.auditor = SenSeIAuditor(distance_x=self.distance_x, distance_y=self.distance_y, num_steps=auditor_nsteps, lr=auditor_lr)

    def train_forward(self, X, Y):
        minlambda = torch.tensor(1.0, device=self.device)
        if self.lamb is None:
            self.lamb = torch.tensor(1.0, device=self.device)
        if type(self.eps) is float:
            self.eps = torch.tensor(self.eps, device=self.device)
        
        Y_pred = self.model(X)
        X_worst = self.auditor.generate_worst_case_examples(self.model, X, lambda_param=self.lamb)

        dist_x = self.distance_x(X, X_worst)
        mean_dist_x = dist_x.mean()
        lr_factor = torch.maximum(mean_dist_x, self.eps) / torch.minimum(mean_dist_x, self.eps)

        self.lamb = torch.max(
            torch.stack(
                [minlambda, self.lamb + lr_factor * (mean_dist_x - self.eps)]
            )
        )

        Y_pred_worst = self.model(X_worst)
        fair_loss = torch.mean(
            self.loss_fn(Y_pred, Y) + self.rho * self.distance_y(Y_pred, Y_pred_worst)
        )

        return fair_loss

    def train(self, patience=20):
        self.model.to(self.device)

        best_ac = None
        counter = 0
        # for e in tqdm(range(self.epochs)):
        for e in range(self.epochs):
            self.model.train()
            for x, y in self.train_dl:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.train_forward(x, y)
                loss.backward()
                self.optimizer.step()
            ac = accuracy(self.model, self.test_dl, self.device)
            if best_ac == None or ac - best_ac > 1e-5:
                best_ac = ac
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break
        print('epoches:', e, '\n')
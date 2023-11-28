import torch
from Unfairness_prove.utils import UnfairMetric
from data.data_utils import DataGenerator
from abc import ABCMeta, abstractmethod

from seeker.seeker import Seeker


class GradiantBasedSeeker(Seeker, metaclass=ABCMeta):
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator):
        super().__init__(model, unfair_metric)
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.sensitiveness = unfair_metric.dx.sensitiveness()

        data_range = self.data_gen.get_range('data')
        self.scale = data_range[1] - data_range[0]

    def _norm(self, x):
        return self.data_gen.norm(x)
    
    def _recover(self, x):
        return self.data_gen.recover(x)
    
    def _query_logits(self, x):
        if len(x.shape) == 1:
            self.n_query += 1
        else:
            self.n_query += x.shape[0]
        x = self._recover(x)
        return self.model(x)
    
    def _query_label(self, x):
        if len(x.shape) == 1:
            self.n_query += 1
        else:
            self.n_query += x.shape[0]
        x = self._recover(x)
        return self.model.get_prediction(x)

    def _check(self, x1, x2, additional_query=0):
        x1, x2 = self._recover(x1), self._recover(x2)
        y1 = self.model.get_prediction(x1)
        y2 = self.model.get_prediction(x2)
        self.n_query += additional_query
        return self.unfair_metric.is_unfair(x1, x2, y1, y2)

    def loss(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        y = self._query_logits(x)[0]
        return y[self.origin_label] - y[self.mis_label]
    
    def reg(self, delta):
        return torch.norm((1 - self.sensitiveness)*delta, p=2)

    @abstractmethod
    def _gradient1(self, x, delta, lamb):
        pass

    def step1(self, x, delta, lr, lamb):
        g = self._gradient1(x, delta, lamb)
        new_delta = delta - lr*g
        new_x = self._norm(self.data_gen.clip(self._recover(x + new_delta)))
        return new_x - x

    @abstractmethod
    def _gradient2(self, x):
        pass

    def after_converge(self, x, max_iter=10):
        for _ in range(max_iter):
            x_recover = self._recover(x)
            g = self._gradient2(x)
            g_direction = torch.sign(g).squeeze()
            x1_candidate = torch.round(self.data_gen.clip(x_recover - torch.diag(g_direction)))
            diff = torch.any(x1_candidate.int() != x_recover.int(), dim=1)
            x1_candidate = x1_candidate[diff]

            distances = self.unfair_metric.dx(x_recover, x1_candidate, itemwise_dist=False).squeeze()
            d_min, idx = torch.min(distances, dim=0)
            if d_min < 1/self.unfair_metric.epsilon:
                x_step = self._norm(x1_candidate[idx]).unsqueeze(0)
                if self._check(x, x_step):
                    pair = self._recover(torch.concat([x, x_step], dim=0))
                    return pair
                else:
                    x = x_step.unsqueeze(0).detach()
            else:
                break
        return None
        
    def seek(self, lamb, origin_lr, max_query):
        self.n_query = 0

        def init():
            x0 = self._norm(self.data_gen.gen_by_range(1))
            delta_t = torch.zeros_like(x0)
            x_t = self._recover(x0)
            pred_t = self._query_label(x0)

            self.origin_label = pred_t.item()
            self.mis_label = 1 - self.origin_label
            lr = origin_lr
            return x0, delta_t, x_t, pred_t, lr
        
        x0, delta_t, x_t, pred_t, lr = init()

        while 1:
            # check if we run out of the query chances
            if self.n_query > max_query:
                return None, self.n_query
            
            # stage 1: fine a x0, which is most likely to have a adversarial
            delta_next = self.step1(x=x0, delta=delta_t, lr=lr, lamb=lamb)
            x_next = self._norm(torch.round(self._recover(x0+delta_next)))
            pred_next = self._query_label(x_next)

            # converage, then stage 2: find an adversarial of x1=(x+delta_t)
            if torch.all(x_next == x_t):
                print('converge:', self.n_query)
                print(self._recover(x0)[0].int(), self._recover(x0)[0][26].int())
                print(self.model(self._recover(x0)))
                print(self._recover(x_t)[0].int(), self._recover(x_t)[0][26].int(), self.n_query)
                print(self.model(self._recover(x_t)))
                x1 = x_t.detach()
                result = self.after_converge(x1)
                if result == None:
                    print('restart')
                    x0, delta_t, x_t, pred_t, lr == init()
                    continue
                return result, self.n_query
            
            if pred_next == self.mis_label:
                print('lr/=5', self.n_query)
                lr /= 5
            else:
                print('continue', self.n_query)
                x_t = x_next
                delta_t = delta_next.detach()

class WhiteboxSeeker(GradiantBasedSeeker):
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator):
        super().__init__(model, unfair_metric, data_gen)
        self.cur_delta = torch.Tensor()
        self.cur_x = torch.Tensor()
        self.cur_g = None

    def _gradient1(self, x, delta, lamb):
        delta.requires_grad=True
        if self.cur_delta.shape[0] != 0 and torch.all(self.cur_delta == delta):
            g = self.cur_g
        else:
            loss = self.loss(x+delta) + lamb * self.reg(delta)
            loss.backward()
            g = delta.grad
            self.cur_delta = delta.clone()
            self.cur_g = g.clone()
        return g

    def _gradient2(self, x):
        x.requires_grad=True
        if self.cur_x.shape[0] != 0 and torch.all(self.cur_x == x):
            g = self.cur_g
        else:
            loss = self.loss(x)
            loss.backward()
            g = x.grad
            self.cur_x = x.clone()
            self.cur_g = g.clone()
        return g
    
class BlackboxSeeker(GradiantBasedSeeker):
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator):
        super().__init__(model, unfair_metric, data_gen)
        self.cur_delta = torch.Tensor()
        self.cur_x = torch.Tensor()
        self.cur_g = None
    
    def _gradient1(self, x, delta, lamb):
        loss = lambda x, delta, lamb: self.loss(x+delta) + lamb * self.reg(delta)

        delta.requires_grad=True
        if self.cur_delta.shape[0] != 0 and torch.all(self.cur_delta == delta):
            g = self.cur_g
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                purt = torch.zeros_like(x).squeeze()
                purt[i] = 1e-3
                purt_delta = [delta + purt, delta - purt]
                g[i] = (loss(x, purt_delta[0], lamb) \
                      - loss(x, purt_delta[1], lamb))/2
            self.cur_delta = delta.clone()
            self.cur_g = g
        return g
    
    def _gradient2(self, x):
        if self.cur_x.shape[0] != 0 and torch.all(self.cur_x == x):
            g = self.cur_g
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                purt = torch.zeros_like(x).squeeze()
                purt[i] = 1
                g[i] = (self.loss(x + purt) - self.loss(x - purt))/2
            self.cur_x = x.clone()
            self.cur_g = g
        return g
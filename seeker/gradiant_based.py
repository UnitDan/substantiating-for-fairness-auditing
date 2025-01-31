import torch
from utils import UnfairMetric
from data.data_utils import DataGenerator
from abc import ABCMeta, abstractmethod
import random

from seeker.seeker import Seeker


class GradiantBasedSeeker(Seeker, metaclass=ABCMeta):
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator):
        super().__init__(model, unfair_metric)
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.sensitiveness = unfair_metric.dx.sensitiveness()

        data_range = self.data_gen.get_range('data')
        self.scale = data_range[1] - data_range[0]

        self.cur_delta = torch.Tensor()
        self.cur_x = torch.Tensor()
        self.cur_g = None

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
        '''
        x1 and x2 should be recovered.
        '''
        # x1, x2 = self._recover(x1), self._recover(x2)
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
        # return torch.norm((1 - self.sensitiveness)*torch.abs(delta), p=2)
        return torch.norm((1 - self.sensitiveness)*delta, p=2)

    @abstractmethod
    def _gradient1(self, x, delta, lamb):
        pass

    def step1(self, x, delta, lr, lamb):
        g = self._gradient1(x, delta, lamb)
        # print('------------g----------------')
        # print(g)
        # print('-----------------------------')
        g = g / torch.norm(g)
        new_delta = delta - lr*g
        new_x = self._norm(self.data_gen.clip(self._recover(x + new_delta)))
        return new_x - x

    @abstractmethod
    def _gradient2(self, x):
        pass

    def _random_perturb(self, x):
        x = x.squeeze()
        pert = torch.eye(x.shape[0])
        pert = torch.concat([pert, -pert])
        x1_candidate = torch.round(self.data_gen.clip(x + pert))
        diff = torch.any(x1_candidate.int() != x.int(), dim=1)
        x1_candidate = x1_candidate[diff]
        return x1_candidate

    def _pert(self, x):
        x_origin = x.clone()

        searched = torch.Tensor()
        while 1:
            x1_candidate = self._random_perturb(x)

            distances = self.unfair_metric.dx(x_origin, x1_candidate, itemwise_dist=False).squeeze()
            x1_candidate = x1_candidate[distances < 1/self.unfair_metric.epsilon]

            set1 = {tuple(row.numpy()) for row in searched.int()}
            set2 = {tuple(row.numpy()) for row in x1_candidate.int()}
            if len(set1) == 0 and len(set2) == 0:
                raise Exception(f'data sample has no possible perturbation within dx <= 1/epsilon ({1/self.unfair_metric.epsilon})')
            
            x1_candidate = torch.Tensor(list(set2 - set1))
            if x1_candidate.shape[0] == 0:
                break

            pert_data = x1_candidate[random.randint(0, x1_candidate.shape[0] - 1)]
            searched = torch.concat([searched, pert_data.unsqueeze(0)], dim=0)

            if random.random() > 0.5:
                break
        return pert_data
     
    def after_converge(self, x, max_iter=10):
        x_recover = torch.round(self._recover(x))
        x_recover0 = x_recover.clone()
        g = None
        for _ in range(max_iter):
            
            if g == None:
                g = self._gradient2(x_recover)
                g_direction = torch.sign(g).squeeze()
            if torch.all(g_direction == 0):
                # print('g == 0')
                return None
            x1_candidate = torch.round(self.data_gen.clip(x_recover - torch.diag(g_direction)))
            diff = torch.any(x1_candidate.int() != x_recover0.int(), dim=1)
            x1_candidate = x1_candidate[diff]
            # print(diff)
            # print(x1_candidate)

            distances = self.unfair_metric.dx(x_recover0, x1_candidate, itemwise_dist=False).squeeze()
            d_min, idx = torch.min(distances, dim=0)
            if d_min < 1/self.unfair_metric.epsilon:
                x_step = x1_candidate[idx].unsqueeze(0)
                # print('x_step after converge\n', x_step)
                if self._check(x_recover0, x_step, 1):
                    pair = torch.concat([x_recover0, x_step], dim=0)
                    return pair
                else:
                    x_recover = x_step.detach()
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
            self.n_iters = 0

            self.cur_delta = torch.Tensor()
            self.cur_x = torch.Tensor()
            self.cur_g = None
            return x0, delta_t, x_t, pred_t, lr
        
        x0, delta_t, x_t, pred_t, lr = init()
        # print(x0)

        while 1:
            # check if we run out of the query chances
            if self.n_query > max_query:
                return None, self.n_query
            
            # stage 1: fine a x0, which is most likely to have a adversarial
            delta_next = self.step1(x=x0, delta=delta_t, lr=lr, lamb=lamb)
            # print('length of the perturbation of delta:', torch.norm(delta_next - delta_t, p=2), sep='\n')
            x_next = self._norm(torch.round(self._recover(x0+delta_next)))
            pred_next = self._query_logits(x_next)[0]
            # print(pred_next)

            # converage, then stage 2: find an adversarial of x1=(x+delta_t)
            if torch.all(x_next == x_t):
                # print('converge', 'n_query:', self.n_query, 'n_iters', self.n_iters)
                # print(self._recover(x0)[0].int())
                # print(self.model(self._recover(x0)))
                # print(self._recover(x_t)[0].int(), self.n_query)
                # print(self.model(self._recover(x_t)))
                # exit()
                x1 = x_t.detach()
                result = self.after_converge(x1)
                if result == None:
                    # print('restart', self.n_query)
                    x0, delta_t, x_t, pred_t, lr = init()
                    continue
                    # return None, self.n_query, self.n_iters
                return result, self.n_query
            
            if pred_next[self.origin_label] - pred_next[self.mis_label] <= 1e-4:
                # print('lr too large', lr)
                lr /= 2
            else:
                # print('next')
                x_t = x_next
                delta_t = delta_next.detach()
                self.n_iters += 1

class WhiteboxSeeker(GradiantBasedSeeker):
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
            loss = self.loss(self._norm(x))
            loss.backward()
            g = x.grad
            self.cur_x = x.clone()
            self.cur_g = g.clone()
        return g
    
class BlackboxSeeker(GradiantBasedSeeker):
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator, easy=True):
        super().__init__(model, unfair_metric, data_gen)
        self.easy = easy
    
    def _gradient1(self, x, delta, lamb):
        loss = lambda x, delta, lamb: self.loss(x+delta) + lamb * self.reg(delta)

        delta.requires_grad=True
        if self.cur_delta.shape[0] != 0 and (torch.all(self.cur_delta == delta) or self.easy):
            # print('old g')
            g = self.cur_g
        else:
            # print(self._recover(x))
            # print('new g')
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = self.g_range
                pert_delta = [delta + pert, delta - pert]
                # print()
                # print(self._recover(x + pert_delta[0]))
                # print(self._recover(x + pert_delta[1]))
                # print()
                g[i] = (loss(x, pert_delta[0], lamb) \
                      - loss(x, pert_delta[1], lamb))/2
                # print(loss(x, pert_delta[0], lamb))
                # print(loss(x, pert_delta[1], lamb))
                # input()
            self.cur_delta = delta.clone()
            self.cur_g = g
            # print(g)
            # exit()
        return g
    
    def _gradient2(self, x):
        self.cur_delta = torch.Tensor()

        if self.cur_x.shape[0] != 0 and torch.all(self.cur_x == x):
            g = self.cur_g
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1
                g[i] = (self.loss(self._norm(x + pert)) - self.loss(self._norm(x - pert)))/2
            self.cur_x = x.clone()
            self.cur_g = g
        return g
    
class FoolSeeker(BlackboxSeeker):
    def _gradient1(self, x, delta, lamb):
        if self.cur_g != None and self.easy:
            g = self.cur_g
        else:
            g = torch.rand_like(x).squeeze()
            self.cur_g = g
        return g
    
    def _gradient2(self, x):
        self.cur_delta = torch.Tensor()

        if self.cur_x.shape[0] != 0 and torch.all(self.cur_x == x):
            g = self.cur_g
        else:
            g = torch.rand_like(x).squeeze()
            self.cur_g = g
        return g
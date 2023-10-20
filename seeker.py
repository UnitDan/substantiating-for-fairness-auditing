import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
import itertools
import random
from utils import Unfair_metric
from data.data_utils import Data_gen
from tqdm import tqdm

class Seeker():
    def __init__(self, model, unfair_metric: Unfair_metric):
        self.model = model
        self.unfair_metric = unfair_metric
        self.n_query = 0

    def _query_logits(self, x):
        if len(x.shape) == 1:
            self.n_query += 1
        else:
            self.n_query += x.shape[0]
        return self.model(x)
    
    def _query_label(self, x):
        if len(x.shape) == 1:
            self.n_query += 1
        else:
            self.n_query += x.shape[0]
        return self.model.get_prediction(x)

    def seek(self):
        pass

class Random_select_seeker(Seeker):
    def __init__(self, model, unfair_metric, data):
        self.data = data
        
        super().__init__(model, unfair_metric)

    def seek(self, dx_constraint=False, max_query=1e3):
        self.n_query = 0
        pair = None

        while self.n_query < max_query:
            if not dx_constraint:
                idx1, idx2 = random.randint(0, self.data.shape[0]-1), random.randint(0, self.data.shape[0]-1)
                x1, x2 = self.data[idx1], self.data[idx2]
            else:
                idx1 = random.randint(0, self.data.shape[0]-1)
                x1 = self.data[idx1]
                distances = self.unfair_metric.dx(x1.unsqueeze(dim=0), self.data, itemwise_dist=False).squeeze()
                id_to_choose = torch.arange(self.data.shape[0])[distances < 1/self.unfair_metric.epsilon]
                if id_to_choose.shape[0] == 0:
                    continue
                idx2 = id_to_choose[torch.randperm(id_to_choose.shape[0])[0]]
                x2 = self.data[idx2]
            
            y1, y2 = self._query_label(x1), self._query_label(x2)
            if self.unfair_metric.is_unfair(x1, x2, y1, y2):
                pair = torch.concat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
                break
        return pair, self.n_query


class Random_gen_seeker(Seeker):
    def __init__(self, model, unfair_metric, data_gen: Data_gen):
        self.data_gen = data_gen
        super().__init__(model, unfair_metric)

    def seek(self, by_range: bool, max_query=1e3):
        self.n_query = 0
        pair = None

        while self.n_query < max_query:
            if by_range:
                x0 = self.data_gen.gen_by_range()
            else:
                x0 = self.data_gen.gen_by_distribution()
            x1_candidate = self.data_gen.random_perturb(x0, dx=self.unfair_metric.dx, epsilon=self.unfair_metric.epsilon)
            x1 = x1_candidate[random.randint(0, x1_candidate.shape[0] - 1)].unsqueeze(0)
            t_pair = torch.concat([x0, x1], dim=0)
            output = self._query_label(t_pair)
            if self.unfair_metric.is_unfair(t_pair[0], t_pair[1], output[0], output[1]):
                pair = t_pair
                break
        return pair, self.n_query
    

                
class White_seeker(Seeker):
    def __init__(self, model, unfair_metric, data_gen: Data_gen):
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()
        super().__init__(model, unfair_metric)

    def step(self, x, lr):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            y0 = self._query_logits(x)[0]
            # loss to minimize
            mis_classification_loss = y0[self.origin_label] - y0[self.mis_label]
            mis_classification_loss.backward()
            g = x.grad
            self.cur_x0 = x.clone()
            self.cur_gradient = g
        return self.data_gen.clip(x - lr*g)

    def seek(self, origin_lr=1e15, max_query=1e3):
        self.n_query = 0

        x0 = self.data_gen.gen_by_range(1)
        x0.requires_grad = True

        with torch.no_grad():
            self.origin_label = self.model.get_prediction(x0).item()
            self.mis_label = 1 - self.origin_label
        lr = origin_lr

        pair = None

        while 1:
            print(lr)
            if x0.grad != None:
                x0.grad.zero_()

            if self.n_query > max_query:
                print('out1: query too many times')
                return None, self.n_query
            x_new = self.step(x0, lr)
            if torch.all(x_new == x0):
                print(f'out4: lr={lr}, x0 convergence')
                # print(x0.int())
                delta = self.unfair_metric.dx.adjust_length(x0, x0.grad, 1/self.unfair_metric.epsilon)
                data_around = self.data_gen.data_around(x0+delta)
                print('len of data_around before filter:', len(data_around))
                distances = self.unfair_metric.dx(data_around, x0, itemwise_dist=False)
                id_to_choose = torch.where(distances < 1/self.unfair_metric.epsilon)[0]
                data_around = data_around[id_to_choose]
                print('len of data_around after filter:', len(data_around))

                for x1 in data_around:
                    x1 = x1.unsqueeze(0)
                    t_pair = torch.concat([x0, x1], dim=0)
                    output = self._query_label(t_pair)
                    if self.unfair_metric.is_unfair(t_pair[0], t_pair[1], output[0], output[1]):
                        pair = t_pair
                        return pair, self.n_query
                x0 = self.data_gen.gen_by_range(1)
                x0.requires_grad = True
                continue
            while self._query_label(x_new) == self.mis_label:
                if self.unfair_metric.dx(x0, x_new) <= 1/self.unfair_metric.epsilon:
                    pair = torch.concat([x0, x_new], dim=0)
                    print('out5: directly find a unfair pair when processing gradiant descent')
                    return pair, self.n_query
                else:
                    if self.n_query >= max_query:
                        print('out2: query too many times')
                        return None, self.n_query
                    
                    while 1:
                        lr /= 5
                        if torch.any(x_new != self.step(x0, lr)):
                            # if can generate a new sample with smaller perturbation, then do it
                            x_new = self.step(x0, lr)
                            break
                        elif torch.all(x0 == self.step(x0, lr)):
                            print('out3: cannot generate a new sample with a smaller perturbation')
                            return None, self.n_query
            print('x0 changes')
            x0 = x_new.detach()
            x0.requires_grad = True

class Black_seeker(Seeker):
    def __init__(self, model, unfair_metric: Unfair_metric, data_gen: Data_gen):
        super().__init__(model, unfair_metric)
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()
    
    def _probability_by_dx(self, x):
        '''
        return the probability of choosing a idx i according to dx.
        p[i] = 1/(1+exp(dx(x+purt_i, x-purt_i)))
        '''
        x = x.squeeze()
        dx = self.unfair_metric.dx
        g = torch.zeros_like(x)
        for i in range(g.shape[0]):
            purt = torch.zeros_like(x)
            purt[i] = 1
            g[i] = -dx(x+purt, x-purt)
        return torch.sigmoid(g)
    
    def loss(self, x):
        y = self._query_logits(x)[0]
        loss = y[self.origin_label] - y[self.mis_label]
        # print(loss)
        return loss

    def _check_dim_and_delta(self, x, i, d):
        dim_min, dim_max = self.data_gen.data_range[0][i], self.data_gen.data_range[1][i]
        if (x[0][i] == dim_max and d >= 0) or (x[0][i] == dim_min and d <= 0):
            return False
        else:
            return True

    def step(self, x, lr):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                purt = torch.zeros_like(x).squeeze()
                purt[i] = 1
                g[i] = (self.loss(x + purt) - self.loss(x - purt))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g
        return self.data_gen.clip(x-lr*g)

    def seek(self, origin_lr=1e15, max_query=1e3):
        self.n_query = 0

        x0 = self.data_gen.gen_by_range(1)

        self.origin_label = self.model.get_prediction(x0).item()
        self.mis_label = 1 - self.origin_label
        lr = origin_lr

        pair = None

        step = 0
        while 1:
            print(step)
            step += 1
            if self.n_query > max_query:
                print('out1: query too many times')
                return None, self.n_query
            x_new = self.step(x0, lr)
            if torch.all(x_new == x0):
                print(f'out4: lr={lr}, x0 convergence')
                print(x0.int())
                delta = self.unfair_metric.dx.adjust_length(x0, self.cur_gradient, 1/self.unfair_metric.epsilon)
                data_around = self.data_gen.data_around(x0+delta)
                print('len of data_around before filter:', len(data_around))
                distances = self.unfair_metric.dx(data_around, x0, itemwise_dist=False)
                id_to_choose = torch.where(distances < 1/self.unfair_metric.epsilon)[0]
                data_around = data_around[id_to_choose]
                print('len of data_around after filter:', len(data_around))
                for x1 in data_around:
                    x1 = x1.unsqueeze(0)
                    t_pair = torch.concat([x0, x1], dim=0)
                    output = self._query_label(t_pair)
                    if self.unfair_metric.is_unfair(t_pair[0], t_pair[1], output[0], output[1]):
                        pair = t_pair
                        return pair, self.n_query
                x0 = self.data_gen.gen_by_range(1)
                lr = origin_lr
                continue
            while self._query_label(x_new) == self.mis_label:
                if self.unfair_metric.dx(x0, x_new) <= 1/self.unfair_metric.epsilon:
                    pair = torch.concat([x0, x_new], dim=0)
                    print('out5: directly find a unfair pair when processing gradiant descent')
                    return pair, self.n_query
                else:
                    if self.n_query >= max_query:
                        print('out2: query too many times')
                        return None, self.n_query
                    
                    while 1:
                        lr /= 5
                        if torch.any(x_new != self.step(x0, lr)):
                            # if can generate a new sample with smaller perturbation, then do it
                            x_new = self.step(x0, lr)
                            break
                        elif torch.all(x0 == self.step(x0, lr)):
                            print('out3: cannot generate a new sample with a smaller perturbation')
                            return None, self.n_query
            x0 = x_new.detach()

    # def seek(self, origin_lr=1000, max_query=1e3):
    #     self.n_query = 0

    #     x0 = self.data_gen.gen_by_range(1)

    #     self.M = torch.zeros_like(x0).squeeze()
    #     self.v = torch.zeros_like(x0).squeeze()
    #     self.T = torch.zeros_like(x0).squeeze()

    #     self.origin_label = self.model.get_prediction(x0).item()
    #     self.mis_label = 1 - self.origin_label
    #     lr = origin_lr

    #     pair = None

    #     while 1:
    #         if self.n_query > max_query:
    #             print('out1: query too many times')
    #             return None, self.n_query
            
    #         try:
    #             x_new = self.step(x0, lr)
    #             # print('x0:', x0)
    #             # print('x_new:', x_new)
    #         except Exception as e:
    #             raise e
    #             return None, self.n_query
            
    #         if torch.all(x_new == x0):
    #             print(f'out4: lr={lr}, x0 convergence')
    #             return None, self.n_query

    #         while self._query_label(x_new) == self.mis_label:
    #             print('touch the boundery')
    #             if self.unfair_metric.dx(x0, x_new) <= self.unfair_metric.epsilon:
    #                 pair = torch.concat(x0, x_new, dim=1)
    #                 return pair, self.n_query
    #             else:
    #                 if self.n_query >= max_query:
    #                     print('out2: query too many times')
    #                     return None, self.n_query
                    
    #                 while 1:
    #                     lr /= 10
    #                     try:
    #                         tx = self.step(x0, lr)
    #                     except Exception as e:
    #                         raise e
    #                         return None, self.n_query
                        
    #                     if torch.any(x_new != tx):
    #                         # if can generate a new sample with smaller perturbation, then do it
    #                         x_new = self.step(x0, lr)
    #                         break
    #                     elif torch.all(x0 == tx):
    #                         print('out3: cannot generate a new sample with a smaller perturbation')
    #                         return None, self.n_query
    #         x0 = x_new.detach()
    #         x0.requires_grad = True

    # def step(self, x, lr):
    #     # randomly choose a coordinate i
    #     i_list = torch.multinomial(self._probability_by_dx(x), x.shape[1], replacement=False)
    #     for idx in range(i_list.shape[0]):
    #         print('---------------------------------------------------')
    #         i = i_list[idx]

    #         purt = torch.zeros_like(x).squeeze()
    #         purt[i] = 1
    #         g_i = (self.loss(x + purt) - self.loss(x - purt))/2

    #         self.T[i] += 1
    #         self.M[i] = self.beta1*self.M[i] + (1 - self.beta1)*g_i
    #         self.v[i] = self.beta2*self.v[i] + (1 - self.beta2)*(g_i**2)
    #         self.M[i] /= (1 - self.beta1**self.T[i])
    #         self.v[i] /= (1 - self.beta2**self.T[i])

    #         d = -lr*self.M[i]/(torch.sqrt(self.v[i]) + self.eps)
    #         delta = torch.zeros_like(x[0])
    #         delta[i] = d
    #         delta[i] = torch.max(delta[i], torch.tensor(1.1)) if delta[i] > 0 else torch.min(delta[i], torch.tensor(-1.1))
            
    #         if self._check_dim_and_delta(x, i, d):
    #             break
    #         if idx == i_list.shape[0] - 1:
    #             raise Exception('can not find a dim to change')
    #         # print('\n---------------------log:-----------------------------------------------')
    #         # print(x, x[0][i])
    #         # print(i, g_i, delta)
    #         # print()
    #         # print(self.data_gen.data_range[0], self.data_gen.data_range[0][i])
    #         # print(self.data_gen.data_range[1], self.data_gen.data_range[1][i])
    #     return self.data_gen.clip(x+delta)

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
        self.candidate_pairs = None
        
        super().__init__(model, unfair_metric)

    def seek(self, dx_constraint=False, max_query=1e3):
        self.n_query = 0
        pair = None

        if dx_constraint and self.candidate_pairs == None:
            ds = TensorDataset(self.data)
            dl = DataLoader(ds, batch_size=1000, shuffle=False)
            dx_matrix = []
            for b in tqdm(dl):
                dx_batch = self.unfair_metric.dx(b[0], self.data, itemwise_dist=False).squeeze()
                dx_matrix.append(dx_batch)
            dx_matrix = torch.concat(dx_matrix, dim=0)
            self.candidate_pairs = torch.where(dx_matrix<1/self.unfair_metric.epsilon)

        while self.n_query < max_query:
            if not dx_constraint:
                idx1, idx2 = random.randint(0, self.data.shape[0]-1), random.randint(0, self.data.shape[0]-1)
                x1, x2 = self.data[idx1], self.data[idx2]
            else:
                i = random.randint(0, self.candidate_pairs[0].shape[0]-1)
                idx1 = self.candidate_pairs[0][i]
                idx2 = self.candidate_pairs[1][i]
                x1 = self.data[idx1]
                x2 = self.data[idx2]
            
            y1, y2 = self._query_label(x1), self._query_label(x2)
            if self.unfair_metric.is_unfair(x1, x2, y1, y2):
                pair = torch.concat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
                break
            if self.n_query % 10000 == 0:
                print(self.n_query)
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
            x1 = self.data_gen.random_perturb(x0, self.unfair_metric).unsqueeze(0)
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
    # def loss(self, x):
    #     # 试验两个版本
    #     # 1. 按概率或者直接选最大的方向，设置delta，计算新loss
    #     # 2. （仅黑盒版本）增加一个loss，约束梯度方向
    #     #     思路：估计x点的二阶梯度，利用x的二阶梯度代替x周围点的二阶梯度，估计每个周围点的梯度
    #     p = self._probability_by_dx(x)
    #     sensitive_dim = torch.argmax(p)
    #     delta = torch.zeros_like(x).squeeze()
    #     delta[sensitive_dim] = 1
    #     delta.requires_grad = False

    #     x1 = x + delta
    #     y = self._query_logits(x)[0]
    #     y1 = self._query_logits(x1)[0]
    #     print(x, y)
    #     print(x1, y1)
    #     print()
    #     loss = y[self.mis_label] - y[self.origin_label] + y1[self.origin_label] - y[self.mis_label]

    #     return loss

    def step(self, x, lr):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            loss = self.loss(x)
            # print('loss', loss)
            loss.backward()
            g = x.grad
            # print('gradiant', g)
            self.cur_x0 = x.clone()
            self.cur_gradient = g.clone()

        data_range = self.data_gen.get_range('data')
        st = data_range[1] - data_range[0]
        print('step change 1', -lr * g, (-lr * g)[0][26], (-lr * g)[0][33])
        print('step change 2', -lr * st * g, (-lr * st * g)[0][26], (-lr * st * g)[0][33])
        # return self.data_gen.clip(x - lr*g)
        return x - lr*st*g

    def _after_converge_step(self, x):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            # print('calculate gradient')
            loss = self.loss(x)
            # print('loss', mis_classification_loss)
            loss.backward()
            g = x.grad
            self.cur_x0 = x.clone()
            self.cur_gradient = g.clone()
        # print('gradiant:\n', g)
        data_range = self.data_gen.get_range('data')
        # print('data scale:\n', data_range[1] - data_range[0])
        # print()
        g_direction = g/torch.abs(g)
        # print(g_direction)
        x_step = self.data_gen.clip(x + torch.diag(g_direction))
        # print(x_step)
        x_step = x_step[torch.any(x_step!=x, dim=1)]
        # print(x_step)

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            return x_step[idx]
        else:
            return None
        # delta = self.unfair_metric.dx.adjust_length(x0, self.cur_gradient, 1/self.unfair_metric.epsilon)
        # data_around = self.data_gen.data_around(x0+delta)
        # distances = self.unfair_metric.dx(data_around, x0, itemwise_dist=False)
        # id_to_choose = torch.where(distances < 1/self.unfair_metric.epsilon)[0]
        # print(id_to_choose)
        # pred_around = self.model.get_prediction(data_around)
        # print(torch.where(pred_around == self.mis_label))
        # data_around = data_around[id_to_choose]
        # return data_around

    def seek(self, origin_lr=1e20, max_query=1e3):
        self.n_query = 0

        x0 = self.data_gen.gen_by_range(1)
        x0.requires_grad = True

        with torch.no_grad():
            self.origin_label = self.model.get_prediction(x0).item()
            self.mis_label = 1 - self.origin_label
        lr = origin_lr

        while 1:
            # print(lr)
            if x0.grad != None:
                x0.grad.zero_()

            if self.n_query > max_query:
                # print(1)
                return None, self.n_query
            x_new = self.step(x0, lr)
            # print('real step change', x_new - x0, (x_new - x0)[0][26], (x_new - x0)[0][33])
            # raise Exception('debug checkpoint')

            # converge
            if torch.all(x_new == x0):
                # print('converge')
                iii = 0
                while 1:
                    iii += 1
                    # print(iii)
                    if self.n_query > max_query:
                        # print(2)
                        return None, self.n_query
                    
                    x_step = self._after_converge_step(x0)
                    if x_step == None:
                        x0 = self.data_gen.gen_by_range(1)
                        x0.requires_grad = True
                        lr = origin_lr
                        break
                    output1 = self.model.get_prediction(x0)
                    output2 = self.model.get_prediction(x_step)
                    if self.unfair_metric.is_unfair(x0, x_step, output1, output2):
                        pair = torch.concat([x0, x_step.unsqueeze(0)], dim=0)
                        return pair, self.n_query
                    else:
                        # print(x_step)
                        x0 = x_step.detach()
                        x0.requires_grad = True
                continue

                # data_around = self._after_converge(x0)

                # for x1 in data_around:
                #     t_pair = torch.concat([x0, x1.unsqueeze(0)], dim=0)
                #     output = self._query_label(t_pair)
                #     if self.unfair_metric.is_unfair(t_pair[0], t_pair[1], output[0], output[1]):
                #         pair = t_pair
                #         return pair, self.n_query
                # x0 = self.data_gen.gen_by_range(1)
                # x0.requires_grad = True
                # continue

            # not converge
            # x0 have already been queried, so no queries are counted here
            output1 = self.model.get_prediction(x0)
            output2 = self._query_label(x_new)
            if self.unfair_metric.is_unfair(x0, x_new, output1, output2):
                # print('dx', self.unfair_metric.dx(x0, x_new))
                pair = torch.concat([x0, x_new], dim=0)
                # print('out1: directly find a unfair pair when processing gradiant descent')
                return pair, self.n_query
            elif output2 == self.mis_label:
                lr /= 5
                while torch.all(x_new == self.step(x0, lr)):
                    lr /=5
            else:
                x0 = x_new.detach()
                x0.requires_grad = True

class Black_seeker(Seeker):
    def __init__(self, model, unfair_metric: Unfair_metric, data_gen: Data_gen):
        super().__init__(model, unfair_metric)
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()

    def dx_direction(self, x):
        x = x.squeeze()
        dx = self.unfair_metric.dx
        g = torch.zeros_like(x)
        for i in range(g.shape[0]):
            purt = torch.zeros_like(x)
            purt[i] = 1
            g[i] = dx(x+purt, x-purt)
        return g

    def loss0(self, x, g):
        g = F.normalize(g, p=2, dim=0)
        dx_direction = self.dx_direction(x)
        return g@dx_direction

    def test_loss0(self):
        x = self.data_gen.gen_by_range(1)
        g = torch.zeros_like(x).squeeze()
        h = torch.zeros_like(x).squeeze()
        loss_x = self.loss(x)
        for i in range(g.shape[0]):
            purt = torch.zeros_like(x).squeeze()
            purt[i] = 1
            g[i] = (self.loss(x + purt) - self.loss(x - purt))/2
            h[i] = (self.loss(x + purt) - 2*loss_x - self.loss(x - purt))
        
        g0 = torch.zeros_like(x).squeeze()
        for i in range(g.shape[0]):
            purt = torch.zeros_like(x).squeeze()
            purt[i] = 1
            x_purt1, x_purt2 = x + purt, x - purt
        return
        
    def loss(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        y = self._query_logits(x)[0]
        try:
            loss = y[self.origin_label] - y[self.mis_label]
        except Exception as e:
            print(x)
            print(y)
            raise e
        # print(loss)
        return loss

    def step(self, x, lr):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            # print('calculate gradient')
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                purt = torch.zeros_like(x).squeeze()
                purt[i] = 1
                g[i] = (self.loss(x + purt) - self.loss(x - purt))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g
        return self.data_gen.clip(x-lr*g)

    def _after_converge_step(self, x):
        # print('after converge')
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
        g_direction = g/torch.abs(g)
        x_step = self.data_gen.clip(x + torch.diag(g_direction))
        x_step = x_step[torch.any(x_step!=x, dim=1)]

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            # print(d_min, torch.any(x_step[idx]!=x))
            return x_step[idx]
        else:
            return None

        # delta = self.unfair_metric.dx.adjust_length(x0, self.cur_gradient, 1/self.unfair_metric.epsilon)
        # data_around = self.data_gen.data_around(x0+delta)
        # distances = self.unfair_metric.dx(data_around, x0, itemwise_dist=False)
        # id_to_choose = torch.where(distances < 1/self.unfair_metric.epsilon)[0]
        # print(id_to_choose)
        # pred_around = self.model.get_prediction(data_around)
        # print(torch.where(pred_around == self.mis_label))
        # data_around = data_around[id_to_choose]
        # return data_around

    def seek(self, origin_lr=1e15, max_query=1e3):
        self.n_query = 0

        x0 = self.data_gen.gen_by_range(1)

        self.origin_label = self.model.get_prediction(x0).item()
        self.mis_label = 1 - self.origin_label
        lr = origin_lr

        while 1:
            if self.n_query > max_query:
                return None, self.n_query
            x_new = self.step(x0, lr)

            # converge
            if torch.all(x_new == x0):
                while 1:
                    if self.n_query > max_query:
                        # print(2)
                        return None, self.n_query
                    
                    x_step = self._after_converge_step(x0)
                    if x_step == None:
                        x0 = self.data_gen.gen_by_range(1)
                        lr = origin_lr
                        break
                    output1 = self.model.get_prediction(x0)
                    output2 = self.model.get_prediction(x_step)
                    if self.unfair_metric.is_unfair(x0, x_step, output1, output2):
                        pair = torch.concat([x0, x_step.unsqueeze(0)], dim=0)
                        return pair, self.n_query
                    else:
                        # print(x_step)
                        # print(x_step == x0)
                        x0 = x_step
                continue

            # not converge
            # x0 have already been queried, so no queries are counted here
            output1 = self.model.get_prediction(x0)
            output2 = self._query_label(x_new)
            if self.unfair_metric.is_unfair(x0, x_new, output1, output2):
                pair = torch.concat([x0, x_new], dim=0)
                # print('out1: directly find a unfair pair when processing gradiant descent')
                return pair, self.n_query
            elif output2 == self.mis_label:
                lr /= 5
                while torch.all(x_new == self.step(x0, lr)):
                    lr /=5
            else:
                x0 = x_new
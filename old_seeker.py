import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
import itertools
import random
from utils import UnfairMetric
from data.data_utils import DataGenerator
from tqdm import tqdm
import sys

class Seeker():
    def __init__(self, model, unfair_metric: UnfairMetric):
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

    def check(self, x1, x2, additional_query=0):
        y1 = self.model.get_prediction(x1)
        y2 = self.model.get_prediction(x2)
        self.n_query += additional_query
        return self.unfair_metric.is_unfair(x1, x2, y1, y2)

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
            
            if self.check(x1, x2, additional_query=2):
                pair = torch.concat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
                break
            if self.n_query % 10000 == 0:
                print(self.n_query)
        return pair, self.n_query


class Random_gen_seeker(Seeker):
    def __init__(self, model, unfair_metric, data_gen: DataGenerator):
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
            if self.check(t_pair[0], t_pair[1], additional_query=2):
                pair = t_pair
                break
        return pair, self.n_query
    
class Random_gen_gradiant_seeker(Seeker):
    def __init__(self, model, unfair_metric, data_gen: DataGenerator):
        self.data_gen = data_gen
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()
        super().__init__(model, unfair_metric)

    def loss(self, x):
        y = self._query_logits(x)[0]
        loss = y[self.origin_label] - y[self.mis_label]
        # print(loss)
        return loss
    
    def step(self, x):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1
                g[i] = (self.loss(x + pert) - self.loss(x - pert))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g

        g_direction = torch.sign(g)
        x_step = torch.round(self.data_gen.clip(x - torch.diag(g_direction)))
        x_step = x_step[torch.any(x_step.int()!=x.int(), dim=1)]

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            return x_step[idx].unsqueeze(0)
        else:
            return None
        
    def seek(self, by_range: bool, max_query=1e3):
        self.n_query = 0
        pair = None

        while self.n_query < max_query:
            if by_range:
                x0 = self.data_gen.gen_by_range()
            else:
                x0 = self.data_gen.gen_by_distribution()
            
            pred_t = self._query_label(x0)
            self.origin_label = pred_t.item()
            self.mis_label = 1 - self.origin_label

            x1 = self.step(x0)
            while x0 != None and torch.any(x0.int() != x1.int()):
                t_pair = torch.concat([x0, x1], dim=0)
                if self.check(t_pair[0], t_pair[1], additional_query=2):
                    return pair, self.n_query
                x1 = self.step(x0)

        return pair, self.n_query
    
class White_seeker(Seeker):
    def __init__(self, model, unfair_metric, data_gen: DataGenerator):
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()
        super().__init__(model, unfair_metric)

    def _probability_by_dx(self, x):
        '''
        return the probability of choosing a idx i according to dx.
        p[i] = 1/(1+exp(dx(x+_i, x-_i)))
        '''
        x = x.squeeze()
        dx = self.unfair_metric.dx
        g = torch.zeros_like(x)
        for i in range(g.shape[0]):
            pert = torch.zeros_like(x)
            pert[i] = 1
            g[i] = -dx(x+pert, x-pert)
        return torch.sigmoid(g)

    def loss(self, x):
        y = self._query_logits(x)[0]
        loss = y[self.origin_label] - y[self.mis_label]
        # print(loss)
        return loss

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
        return self.data_gen.clip(x - lr*st*g)

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
        x_step = torch.round(self.data_gen.clip(x + torch.diag(g_direction)))
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
                    if self.check(x0, x_step):
                        pair = torch.concat([x0, x_step.unsqueeze(0)], dim=0)
                        return pair, self.n_query
                    else:
                        # print(x_step)
                        x0 = x_step.detach()
                        x0.requires_grad = True
                continue

            # not converge
            # x0 have already been queried, so no queries are counted here
            output2 = self._query_label(x_new)
            if self.check(x0, x_new):
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
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator):
        super().__init__(model, unfair_metric)
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()

    def loss(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        y = self._query_logits(x)[0]
        loss = y[self.origin_label] - y[self.mis_label]
        # print(loss)
        return loss

    def step(self, x, lr):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            # print('calculate gradient')
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1
                g[i] = (self.loss(x + pert) - self.loss(x - pert))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g
        return torch.round(self.data_gen.clip(x-lr*g))

    def _after_converge_step(self, x):
        # print('after converge')
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1
                g[i] = (self.loss(x + pert) - self.loss(x - pert))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g
        g_direction = g/torch.abs(g)
        x_step = torch.round(self.data_gen.clip(x + torch.diag(g_direction)))
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
                    if self.check(x0, x_step):
                        pair = torch.concat([x0, x_step.unsqueeze(0)], dim=0)
                        return pair, self.n_query
                    else:
                        # print(x_step)
                        # print(x_step == x0)
                        x0 = x_step
                continue

            # not converge
            # x0 have already been queried, so no queries are counted here
            output2 = self._query_label(x_new)
            if self.check(x0, x_new):
                pair = torch.concat([x0, x_new], dim=0)
                # print('out1: directly find a unfair pair when processing gradiant descent')
                return pair, self.n_query
            elif output2 == self.mis_label:
                lr /= 5
                while torch.all(x_new == self.step(x0, lr)):
                    lr /=5
            else:
                x0 = x_new

class Test_seeker(Seeker):
    def __init__(self, model, unfair_metric: UnfairMetric, data_gen: DataGenerator):
        super().__init__(model, unfair_metric)
        self.data_gen = data_gen
        self.origin_label, self.mis_label = None, None
        self.cur_gradient = torch.Tensor()
        self.cur_x0 = torch.Tensor()
        self.sensitiveness = unfair_metric.dx.sensitiveness()

        data_range = self.data_gen.get_range('data')
        self.scale = data_range[1] - data_range[0]

    def loss(self, x):
        y = self._query_logits(x)[0]
        loss = y[self.origin_label] - y[self.mis_label]
        # print(loss)
        return loss

    def delta_init(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        def loss(x):
            y = self._query_logits(x)[0]
            return y[self.origin_label] - y[self.mis_label]
        
        g = torch.zeros_like(x).squeeze()
        for i in range(g.shape[0]):
            pert = torch.zeros_like(x).squeeze()
            pert[i] = 1
            g[i] = torch.sign(loss(x + pert) - loss(x - pert))
        return g

    def loss1(self, x, delta, lamb):
        '''
        x是x0+delta_t
        delta是delta_t或者delta_t - delta_(t-1)
        '''
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        y = self._query_logits(x)[0]

        # print(delta, delta[0][26])
        # loss = y[self.origin_label] - y[self.mis_label]
        # loss.backward(retain_graph=True)
        # print('loss grad', delta.grad[0], delta.grad[0][26])
        # delta.grad.zero_()

        # reg = torch.norm((1 - self.sensitiveness)*delta, p=2)
        # reg.backward(retain_graph=True)
        # print('reg grad', delta.grad[0], delta.grad[0][26])
        # delta.grad.zero_()

        loss = y[self.origin_label] - y[self.mis_label]
        reg = torch.norm((1 - self.sensitiveness)*delta, p=2)
        # print('loss', loss.item(), 'reg', reg.item(), '\n')
        return loss + lamb * reg
    
    def step1_white(self, x, delta, lr, lamb):
        if delta.grad == None:
            # loss = self.loss1(x+delta, delta, lamb)
            loss = self.loss1(x+delta, delta - delta.detach(), lamb)
            # print('loss', loss)
            loss.backward()
        g = delta.grad
        
        # 两种scale-norm的方式
        new_delta = delta - lr*self.scale*g
        print('-------------------------------- delta update --------------------------------------')
        print('origin\n', delta, delta[0][26])
        print('step size\n', (-lr*self.scale*g)[0], (-lr*self.scale*g)[0][26])
        print('now\n', new_delta, new_delta[0][26])
        print('------------------------------------------------------------------------------------\n\n')
        # new_delta = delta - lr*g
        new_x = self.data_gen.clip(x + new_delta)
        return new_x - x

    def step1_black(self, x, delta, lr, lamb):
        g = torch.zeros_like(x).squeeze()
        for i in range(g.shape[0]):
            pert = torch.zeros_like(x).squeeze()
            # pert[i] = self.scale[i]
            pert[i] = 1
            pert_delta = [delta + pert, delta - pert]
            # v1: 限制相对于x0的总变化量，应当尽量少在敏感属性上发生变化
            g[i] = (self.loss1(x+pert_delta[0], pert_delta[0], lamb) - self.loss1(x+pert_delta[1], pert_delta[1]))/2
            # v2：限制当前单步的变化量，应当尽量少在敏感属性上发生变化
            # g[i] = (self.loss1(x+pert_delta[0], pert, lamb) - self.loss1(x+pert_delta[1], -pert))/2
        
        # 两种scale-norm的方式
        new_delta = delta - lr*self.scale*g
        # new_delta = delta - lr*g
        new_x = self.data_gen.clip(x + new_delta)
        return new_x - x

    def loss2(self, x, delta, lamb):
        x1 = x + delta
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        
        y = self._query_label(x1)[0]
        loss = y[self.origin_label] - y[self.mis_label]
        reg = torch.norm(self.sensitiveness*delta, p=2)
        return loss + lamb * reg
    
    def _after_converge_step_white(self, x):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            loss = self.loss(x)
            loss.backward()
            g = x.grad
            self.cur_x0 = x.clone()
            self.cur_gradient = g.clone()

        g_direction = torch.sign(g)
        x_step = torch.round(self.data_gen.clip(x + torch.diag(g_direction)))
        x_step = x_step[torch.any(x_step!=x, dim=1)]

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            return x_step[idx]
        else:
            return None

    def _after_converge_step_black(self, x):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1
                g[i] = (self.loss(x + pert) - self.loss(x - pert))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g
        g_direction = g/torch.abs(g)
        x_step = torch.round(self.data_gen.clip(x + torch.diag(g_direction)))
        x_step = x_step[torch.any(x_step!=x, dim=1)]

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            return x_step[idx]
        else:
            return None

    def seek(self, black_box, lamb=1, origin_lr=1e5, max_query=1e3):
        self.n_query = 0
        def init():
            x0 = self.data_gen.gen_by_range(1)
            delta_t = torch.zeros_like(x0)
            if not black_box:
                delta_t.requires_grad = True

            x_t = torch.round(x0+delta_t)
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
            
            # step1: fine a x0, which is most likely to have a adversarial
            if black_box:
                delta_next = self.step1_black(x=x0, delta=delta_t, lr=lr, lamb=lamb)
            else:
                delta_next = self.step1_white(x=x0, delta=delta_t, lr=lr, lamb=lamb)

            x_next = torch.round(x0 + delta_next)
            pred_next = self._query_label(x_next)

            # converage, then step 2: find a adversarial of x1=(x+delta_t)
            if torch.all(x_next == x_t):
                print('converge')
                x1 = torch.round(x0 + delta_t).detach()
                if not black_box:
                    x1.requires_grad = True
                while 1:
                    if self.n_query > max_query:
                        # print(2)
                        return None, self.n_query
                    
                    if black_box:
                        x_step = self._after_converge_step_black(x1)
                    else:
                        x_step = self._after_converge_step_white(x1)
                    if x_step == None:
                        print('\nrestart')
                        x0, delta_t, x_t, pred_t, lr = init()
                        print(x0.int())
                        break
                    if self.check(x1, x_step):
                        pair = torch.concat([x1, x_step.unsqueeze(0)], dim=0)
                        return pair, self.n_query
                    else:
                        x1 = x_step.detach()
                        if not black_box:
                            x1.requires_grad = True
                continue

            if self.check(x_t, x_next):
                # check if we find an unfair pair in step 1
                pair = torch.concat([x_t, x_next], dim=0)
                print(pair.int(), self.n_query)
                # return pair, self.n_query
            elif pred_next == self.mis_label:
                # If the next step will cross the discriminant boundary, 
                # roll back the next step and reduce the step size
                lr /= 5
            else:
                x_t = x_next
                pred_t = pred_next
                delta_t = delta_next.detach()
                if not black_box:
                    delta_t.requires_grad = True

class Norm_Test_seeker(Test_seeker):
    def norm(self, x):
        return self.data_gen.norm(x)
    
    def recover(self, x):
        return self.data_gen.recover(x)

    def _query_logits(self, x):
        if len(x.shape) == 1:
            self.n_query += 1
        else:
            self.n_query += x.shape[0]
        x = self.recover(x)
        return self.model(x)
    
    def _query_label(self, x):
        if len(x.shape) == 1:
            self.n_query += 1
        else:
            self.n_query += x.shape[0]
        x = self.recover(x)
        return self.model.get_prediction(x)

    def check(self, x1, x2, additional_query=0):
        x1, x2 = self.recover(x1), self.recover(x2)
        y1 = self.model.get_prediction(x1)
        y2 = self.model.get_prediction(x2)
        self.n_query += additional_query
        return self.unfair_metric.is_unfair(x1, x2, y1, y2)

    def loss1(self, x, delta, lamb):
        '''
        x是x0+delta_t
        delta是delta_t或者delta_t - delta_(t-1)
        '''
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        y = self._query_logits(x)[0]

        loss = y[self.origin_label] - y[self.mis_label]
        reg = torch.norm((1 - self.sensitiveness)*delta, p=2)
        # print('loss', loss.item(), 'reg', reg.item(), '\n')
        return loss + lamb * reg

    def step1_white(self, x, delta, lr, lamb):
        if delta.grad == None:
            loss = self.loss1(x+delta, delta, lamb)
            # loss = self.loss1(x+delta, delta - delta.detach(), lamb)
            # print('loss', loss)
            loss.backward()
        g = delta.grad

        new_delta = delta - lr*g
        # print('-------------------------------- delta update --------------------------------------')
        # print('origin\n', delta, delta[0][26])
        # print('step size\n', (-lr*g)[0], (-lr*g)[0][26])
        # print('now\n', new_delta, new_delta[0][26])
        # print('------------------------------------------------------------------------------------\n\n')
        # new_delta = delta - lr*g
        new_x = self.norm(self.data_gen.clip(self.recover(x + new_delta)))

        # print('-------------------------------- x update --------------------------------------')
        # print('origin\n', x, x[0][26])
        # print('before clip\n', x + new_delta, (x + new_delta)[0][26])
        # print('before clip recover\n', self.recover(x + new_delta), self.recover(x + new_delta)[0][26])
        # print('after clip recover\n', self.data_gen.clip1(self.recover(x + new_delta)), self.data_gen.clip1(self.recover(x + new_delta))[0][26])
        # print('now\n', new_x, new_x[0][26])
        # print('------------------------------------------------------------------------------------\n\n')

        return new_x - x

    def step1_black(self, x, delta, lr, lamb):
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == delta):
            g = self.cur_gradient
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1e-3
                pert_delta = [delta + pert, delta - pert]
                # v1: 限制相对于x0的总变化量，应当尽量少在敏感属性上发生变化
                g[i] = (self.loss1(x+pert_delta[0], pert_delta[0], lamb) - self.loss1(x+pert_delta[1], pert_delta[1], lamb))/2
                # v2：限制当前单步的变化量，应当尽量少在敏感属性上发生变化
                # g[i] = (self.loss1(x+pert_delta[0], pert, lamb) - self.loss1(x+pert_delta[1], -pert))/2
            self.cur_x0 = delta.clone()
            self.cur_gradient = g.clone()
        
        # 两种scale-norm的方式
        new_delta = delta - lr*g
        # new_delta = delta - lr*g
        new_x = self.norm(self.data_gen.clip(self.recover(x + new_delta)))
        return new_x - x

    def loss(self, x):
        y = self._query_logits(self.norm(x))[0]
        loss = y[self.origin_label] - y[self.mis_label]
        # print(loss)
        return loss

    def _after_converge_step_white(self, x):
        x = self.recover(x).detach()
        x.requires_grad=True
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            loss = self.loss(x)
            loss.backward()
            g = x.grad
            self.cur_x0 = x.clone()
            self.cur_gradient = g.clone()

        g_direction = torch.sign(g).squeeze()
        x_step = torch.round(self.data_gen.clip(x - torch.diag(g_direction)))
        x_step = x_step[torch.any(x_step.int()!=x.int(), dim=1)]

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            return self.norm(x_step[idx])
        else:
            return None

    def _after_converge_step_black(self, x):
        x = self.recover(x)
        if self.cur_x0.shape[0] != 0 and torch.all(self.cur_x0 == x):
            g = self.cur_gradient
        else:
            g = torch.zeros_like(x).squeeze()
            for i in range(g.shape[0]):
                pert = torch.zeros_like(x).squeeze()
                pert[i] = 1
                g[i] = (self.loss(x + pert) - self.loss(x - pert))/2
            self.cur_x0 = x.clone()
            self.cur_gradient = g

        g_direction = torch.sign(g)
        x_step = torch.round(self.data_gen.clip(x - torch.diag(g_direction)))
        x_step = x_step[torch.any(x_step.int()!=x.int(), dim=1)]

        distances = self.unfair_metric.dx(x, x_step, itemwise_dist=False).squeeze()
        d_min, idx = torch.min(distances, dim=0)
        if d_min < 1/self.unfair_metric.epsilon:
            return self.norm(x_step[idx])
        else:
            return None

    def loss2(self, x, delta, lamb):
        x1 = x + delta
        
        y = self._query_logits(x1)[0]

        print(delta, delta[0][26])
        loss = y[self.origin_label] - y[self.mis_label]
        loss.backward(retain_graph=True)
        print('loss grad', delta.grad[0], delta.grad[0][26])
        delta.grad.zero_()

        reg = torch.norm((1 - self.sensitiveness)*delta, p=2)
        reg.backward(retain_graph=True)
        print('reg grad', delta.grad[0], delta.grad[0][26])
        delta.grad.zero_()

        loss = y[self.origin_label] - y[self.mis_label]
        reg = torch.norm(self.sensitiveness*delta, p=2)
        # reg = self.unfair_metric.dx(x, x1)
        return loss + lamb * reg

    def step2_white(self, x, delta, lr, lamb):
        if delta.grad == None:
            loss = self.loss2(x+delta, delta, lamb)
            print('loss', loss)
            loss.backward()
        g = delta.grad
        new_delta = delta - lr*g
        new_x = self.norm(self.data_gen.clip(self.recover(x + new_delta)))
        return new_x
    
    def after_converge1(self, x):
        xx = x.clone()
        xx[0][26] = 1 - xx[0][26]
        print(self.recover(xx).int())
        pair = self.recover(torch.concat([x, xx], dim=0))
        if self.check(x, xx):
            return pair, self.n_query
        else:
            return None, self.n_query

    def after_converge2(self, x, black_box, max_query, lr, lamb):
        delta = torch.zeros_like(x).detach()
        if not black_box:
            delta.requires_grad = True
        while 1:
            if self.n_query > max_query:
                # print(2)
                return None, self.n_query
            
            if black_box:
                x_step = self._after_converge_step_black(x)
            else:
                x_step = self.step2_white(x, delta, lr, lamb)

            if self.check(x, torch.round(x_step)):
                pair = self.recover(torch.concat([x, torch.round(x_step.unsqueeze)], dim=0))
                return pair, self.n_query
            else:
                delta = (x_step - x).detach()
                if not black_box:
                    delta.requires_grad = True
                if self.unfair_metric.dx(x_step, x) > 1/self.unfair_metric.epsilon:
                    lamb *= 2

    def after_converge3(self, x, black_box):
        # print('x!', self.recover(x).int())
        if black_box:
            x_step = self._after_converge_step_black(x)
        else:
            x.requires_grad = True
            x_step = self._after_converge_step_white(x)

        if x_step == None:
            print('\nrestart')
            # x0, delta_t, x_t, pred_t, lr = init()
            # print(x0.int())
            # break
            return None, self.n_query
        if self.check(x, x_step):
            pair = self.recover(torch.concat([x, x_step.unsqueeze(0)], dim=0))
            return pair, self.n_query
        else:
            x = x_step.unsqueeze(0).detach()
            try:
                return self.after_converge3(x, black_box)
            except:
                return None, self.n_query

    def seek(self, black_box, lamb1=1, origin_lr1=1e5, max_query=1e3, lr2=0.01, lamb2=1):
        self.n_query = 0
        def init():
            x0 = self.norm(self.data_gen.gen_by_range(1))
            delta_t = torch.zeros_like(x0)
            if not black_box:
                delta_t.requires_grad = True

            x_t = self.recover(x0+delta_t)
            pred_t = self._query_label(x0)
            self.origin_label = pred_t.item()
            self.mis_label = 1 - self.origin_label
            lr = origin_lr1
            return x0, delta_t, x_t, pred_t, lr
        
        x0, delta_t, x_t, pred_t, lr = init()

        while 1:
            # check if we run out of the query chances
            if self.n_query > max_query:
                return None, self.n_query
            
            # step1: fine a x0, which is most likely to have a adversarial
            if black_box:
                delta_next = self.step1_black(x=x0, delta=delta_t, lr=lr, lamb=lamb1)
            else:
                delta_next = self.step1_white(x=x0, delta=delta_t, lr=lr, lamb=lamb1)

            x_next = self.norm(torch.round(self.recover(x0 + delta_next)))
            pred_next = self._query_label(x_next)


            # converage, then step 2: find an adversarial of x1=(x+delta_t)
            if torch.all(x_next == x_t):
                print('converge', self.n_query)
                print(self.recover(x0)[0].int(), self.recover(x0)[0][26].int())
                print(self.model(self.recover(x0)))
                print(self.recover(x_t)[0].int(), self.recover(x_t)[0][26].int(), self.n_query)
                print(self.model(self.recover(x_t)))
                # return None, self.n_query
                x1 = x_t.detach()
                self.cur_x0 = torch.Tensor()
                # print('x1', x1)
                # return self.after_converge1(x1)
                # return self.after_converge2(x1, black_box, max_query, lr2, lamb2)
                result = self.after_converge3(x1, black_box)
                if result[0] == None:
                    x0, delta_t, x_t, pred_t, lr = init()
                    continue
                return result

            if self.check(x_t, x_next):
                print('lr/=5', self.n_query)
                # print('direct unfair')
                # check if we find an unfair pair in step 1
                pair = self.recover(torch.concat([x_t, x_next], dim=0))
                lr /= 5
                # return pair, self.n_query
            elif pred_next == self.mis_label:
                print('lr/=5', self.n_query)
                # print('cross but too far')
                # If the next step will cross the discriminant boundary, 
                # roll back the next step and reduce the step size
                lr /= 5
            else:
                print('continue', self.n_query)
                # print('next step 1')
                x_t = x_next
                delta_t = delta_next.detach()
                if not black_box:
                    delta_t.requires_grad = True
        
            # if input() == 'q':
            #     raise Exception('stop')
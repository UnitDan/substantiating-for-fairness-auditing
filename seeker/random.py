from seeker.seeker import Seeker
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from data.data_utils import DataGenerator
from tqdm import tqdm

class RandomSelectPairSeeker(Seeker):
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
            
            if self._check(x1, x2, additional_query=2):
                pair = torch.concat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
                break
            if self.n_query % 10000 == 0:
                print(self.n_query)
        return pair, self.n_query

class RandomSeeker(Seeker):
    def __init__(self, model, unfair_metric, data_gen: DataGenerator):
        self.data_gen = data_gen
        super().__init__(model, unfair_metric)

    def seek(self, max_query=1e3):
        self.n_query = 0
        pair = None

        while self.n_query < max_query:
            x0 = self.gen_x0()
            if x0 == None:
                break
            x1 = self.data_gen.random_perturb(x0, self.unfair_metric).unsqueeze(0)
            t_pair = torch.concat([x0, x1], dim=0)
            if self._check(t_pair[0], t_pair[1], additional_query=2):
                pair = t_pair
                break
        return pair, self.n_query

class RandomSelectSeeker(RandomSeeker):
    def __init__(self, model, unfair_metric, data):
        ds = TensorDataset(data)
        dl = DataLoader(ds, batch_size=1, shuffle=True)
        self.data_iter = iter(dl)
        Seeker.__init__(self, model, unfair_metric)
    
    def gen_x0(self):
        try:
            return next(self.data_iter)
        except:
            return None

class RangeGenSeeker(RandomSeeker):
    def __init__(self, model, unfair_metric, data_gen: DataGenerator):
        self.gen_x0 = self.data_gen.gen_by_range
        super().__init__(model, unfair_metric, data_gen)

class DistributionGenSeeker(RandomSeeker):
    def __init__(self, model, unfair_metric, data_gen: DataGenerator):
        self.gen_x0 = self.data_gen.gen_by_distribution
        super().__init__(model, unfair_metric, data_gen) 
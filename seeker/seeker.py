from utils import UnfairMetric
from abc import ABCMeta, abstractmethod

class Seeker(metaclass=ABCMeta):
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

    def _check(self, x1, x2, additional_query=0):
        y1 = self.model.get_prediction(x1)
        y2 = self.model.get_prediction(x2)
        self.n_query += additional_query
        return self.unfair_metric.is_unfair(x1, x2, y1, y2)

    @abstractmethod
    def seek(self):
        pass
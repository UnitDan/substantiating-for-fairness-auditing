import torch
import numpy as np

from distances.distance import Distance
from torch import vmap


class MahalanobisDistances(Distance):
    def __init__(self):
        super().__init__()

        self.device = torch.device("cpu")
        self._vectorized_dist = None
        self.register_buffer("sigma", torch.Tensor())

        self.max = None
        self.min = None

    def to(self, device):
        assert (
            self.sigma is not None and len(self.sigma.size()) != 0
        ), "Please fit the metric before moving parameters to device"

        self.device = device
        self.sigma = self.sigma.to(self.device)

    def fit(self, num_dims: int, sigma, data_gen):
        self.sigma = sigma

        data_range = data_gen.get_range('data')
        self.max = data_range[1].to(self.device)
        self.min = data_range[0].to(self.device)
        self.num_dims = num_dims

    def __compute_dist__(self, X1, X2):
        """Computes the distance between two data samples x1 and x2

        Parameters
        -----------
            X1: torch.Tensor
                Data sample of shape (n_features) or (N, n_features)
            X2: torch.Tensor
                Data sample of shape (n_features) or (N, n_features)

        Returns:
            dist: torch.Tensor
                Distance between points x1 and x2. Shape: (N)
        """

        # unsqueeze batch dimension if a vector is passed
        if len(X1.shape) == 1:
            X1 = X1.unsqueeze(0)
        if len(X2.shape) == 1:
            X2 = X2.unsqueeze(0)

        def _norm(x):
            return (x - self.min)/(self.max - self.min)

        X1 = _norm(X1)
        X2 = _norm(X2)

        X_diff = X1 - X2
        dist = torch.sum((X_diff @ self.sigma) * X_diff, dim=-1, keepdim=True)
        return dist

    def __init_vectorized_dist__(self):
        """Initializes a vectorized version of the distance computation"""
        if self._vectorized_dist is None:
            self._vectorized_dist = vmap(
                vmap(
                    vmap(self.__compute_dist__, in_dims=(None, 0)),
                    in_dims=(0, None),
                ),
                in_dims=(0, 0),
            )

    def forward(self, X1, X2, itemwise_dist=True):
        """Computes the distance between data samples X1 and X2

        Parameters
        -----------
            X1: torch.Tensor
                Data samples from batch 1 of shape (n_samples_1, n_features)
            X2: torch.Tensor
                Data samples from batch 2 of shape (n_samples_2, n_features)
            itemwise_dist: bool, default: True
                Compute the distance in an itemwise manner or pairwise manner.

                In the itemwise fashion (`itemwise_dist=True`), distance is
                computed between the ith data sample in X1 to the ith data sample
                in X2. Thus, the two data samples X1 and X2 should be of the same shape

                In the pairwise fashion (`itemwise_dist=False`), distance is
                computed between all the samples in X1 and all the samples in X2.
                In this case, the two data samples X1 and X2 can be of different shapes.

        Returns
        ----------
            dist: torch.Tensor
                Distance between samples of batch 1 and batch 2.

                If `itemwise_dist=True`, item-wise distance is returned of
                shape (n_samples, 1)

                If `itemwise_dist=False`, pair-wise distance is returned of
                shape (n_samples_1, n_samples_2)
        """

        X1 = X1.unsqueeze(0) if len(X1.shape) == 1 else X1  # (N, D)
        X2 = X2.unsqueeze(0) if len(X2.shape) == 1 else X2  # (M, D)

        if itemwise_dist:
            np.testing.assert_array_equal(
                X1.shape,
                X2.shape,
                err_msg="X1 and X2 should be of the same shape for itemwise distance computation",
            )
            dist = self.__compute_dist__(X1, X2)
        else:
            self.__init_vectorized_dist__()

            X1 = X1.unsqueeze(0) if len(X1.shape) == 2 else X1  # (B, N, D)
            X2 = X2.unsqueeze(0) if len(X2.shape) == 2 else X2  # (B, M, D)
            
            nsamples_x1 = X1.shape[1]
            nsamples_x2 = X2.shape[1]
            dist_shape = (-1, nsamples_x1, nsamples_x2)

            dist = self._vectorized_dist(X1, X2).view(dist_shape)

        return dist

    def adjust_length(self, X, Delta, goal):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        if len(Delta.shape) == 1:
            Delta = Delta.unsqueeze(0)
        X2 = X + Delta

        cun_lenth = self.__compute_dist__(X, X2)
        scaling_factor = torch.sqrt(cun_lenth / goal)

        return scaling_factor*Delta - (scaling_factor-1)*self.min

    def sensitiveness(self):
        x = torch.zeros(self.num_dims)
        pert = torch.diag(self.max - self.min)
        # pert = torch.diag(torch.ones_like(x))
        g = torch.zeros_like(x)
        for i in range(g.shape[0]):
            g[i] = self.forward(x, x+pert[i])
        g = g / torch.max(g)
        return g
    
class SquaredEuclideanDistance(MahalanobisDistances):
    """
    computes the max-min normalized squared euclidean distance where

    .. math:: \\Sigma= I_{num_dims}
    """

    def fit(self, num_dims: int, data_gen):
        """Fit Square Euclidean Distance metric

        Parameters
        -----------------
            num_dims: int
                the number of dimensions of the space in which the Squared Euclidean distance will be used.
        """
        sigma = torch.eye(num_dims).detach()
        super().fit(num_dims, sigma, data_gen)


class ProtectedSEDistances(SquaredEuclideanDistance):
    '''Compute the Protected Squared Euclidean Distance metric which is similar to Squared Euclidean Distance
     while ignore the sensitive attributes while computing distance between data points.
    '''
    def __init__(self):
        super().__init__()
        self._sensitive_attributes = None

    def fit(self, sensitive_idx, num_dims:int, data_gen):
        """Fit Causal Distance metric

        Parameters
        ------------
            sensitive_attributes: Iterable[int]
                List of attribute indices considered to be sensitive.
                The metric would ignore these attributes while
                computing distance between data points.
            num_dims: int
                Total number of attributes in the data points.
        """
        super().fit(num_dims, data_gen)
        self._sensitive_attributes = sensitive_idx
        for p in self._sensitive_attributes:
            self.sigma[p][p] = 0

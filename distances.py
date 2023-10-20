from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from torch import vmap
from inFairness.distances import LogisticRegSensitiveSubspace

class Distance(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for model distances
    """

    def __init__(self):
        super().__init__()

    def fit(self, **kwargs):
        """
        Fits the metric parameters for learnable metrics
        Default functionality is to do nothing. Subclass
        should overwrite this method to implement custom fit
        logic
        """
        pass

    def load_state_dict(self, state_dict, strict=True):

        buffer_keys = [bufferitem[0] for bufferitem in self.named_buffers()]
        for key, val in state_dict.items():
            if key not in buffer_keys and strict:
                raise AssertionError(
                    f"{key} not found in metric state and strict parameter is set to True. Either set strict parameter to False or remove extra entries from the state dictionary."
                )
            setattr(self, key, val)

    @abstractmethod
    def forward(self, x, y):
        """
        Subclasses must override this method to compute particular distances

        Returns:
             Tensor: distance between two inputs
        """

    def adjust_length(self, X, Delta, goal):
        '''
        Clip (or elongate) the perturbation `delta` 
        at sample `x` in a way that maintains the direction of `delta`, resulting in a vector
        of length `goal` with the distance measure of this class.

        Cannot handle the case that the length of `delta` will never be `goal` without changing
        the direction.
        '''
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        if len(Delta.shape) == 1:
            Delta = Delta.unsqueeze(0)

        def length(x, d):
            x2 = x + d
            return self.forward(x, x2)
        
        scaling_factor = torch.sqrt(length(X, Delta) / goal)
        left, right = scaling_factor/2, scaling_factor*2
        while torch.any(length(X, left*Delta)>goal):
            left[length(X, left*Delta)>goal] /= 2
        while torch.any(length(X, right*Delta)<goal):
            right[length(X, right*Delta)<goal] *=2

        while torch.any(left != right):
            mid = (left + right) // 2
            l = length(X, mid*Delta)
            left[l-goal < 0] = mid
            right[goal-l < 1e-5] = mid

        return mid*Delta


class MahalanobisDistances(Distance):
    """Base class implementing the Generalized Mahalanobis Distances

    Mahalanobis distance between two points X1 and X2 is computed as:

    .. math:: \\text{dist}(X_1, X_2) = (X_1 - X_2) \\Sigma (X_1 - X_2)^{T}
    """

    def __init__(self):
        super().__init__()

        self.device = torch.device("cpu")
        self._vectorized_dist = None
        self.register_buffer("sigma", torch.Tensor())

    def to(self, device):
        """Moves distance metric to a particular device

        Parameters
        ------------
            device: torch.device
        """

        assert (
            self.sigma is not None and len(self.sigma.size()) != 0
        ), "Please fit the metric before moving parameters to device"

        self.device = device
        self.sigma = self.sigma.to(self.device)

    def fit(self, sigma):
        """Fit Mahalanobis Distance metric

        Parameters
        ------------
            sigma: torch.Tensor
                    Covariance matrix
        """

        self.sigma = sigma

    @staticmethod
    def __compute_dist__(X1, X2, sigma):
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

        X_diff = X1 - X2
        dist = torch.sum((X_diff @ sigma) * X_diff, dim=-1, keepdim=True)
        return dist

    def __init_vectorized_dist__(self):
        """Initializes a vectorized version of the distance computation"""
        if self._vectorized_dist is None:
            self._vectorized_dist = vmap(
                vmap(
                    vmap(self.__compute_dist__, in_dims=(None, 0, None)),
                    in_dims=(0, None, None),
                ),
                in_dims=(0, 0, None),
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

                In the itemwise fashion (`itemwise_dist=False`), distance is
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

        if itemwise_dist:
            np.testing.assert_array_equal(
                X1.shape,
                X2.shape,
                err_msg="X1 and X2 should be of the same shape for itemwise distance computation",
            )
            dist = self.__compute_dist__(X1, X2, self.sigma)
        else:
            self.__init_vectorized_dist__()

            X1 = X1.unsqueeze(0) if len(X1.shape) == 2 else X1  # (B, N, D)
            X2 = X2.unsqueeze(0) if len(X2.shape) == 2 else X2  # (B, M, D)
            
            nsamples_x1 = X1.shape[1]
            nsamples_x2 = X2.shape[1]
            dist_shape = (-1, nsamples_x1, nsamples_x2)

            dist = self._vectorized_dist(X1, X2, self.sigma).view(dist_shape)

        return dist
    
    def adjust_length(self, X, Delta, goal):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        if len(Delta.shape) == 1:
            Delta = Delta.unsqueeze(0)
        X2 = X + Delta

        cun_lenth = self.__compute_dist__(X, X2, self.sigma)
        scaling_factor = torch.sqrt(cun_lenth / goal)
        return scaling_factor*Delta

        
class NormalizedSquaredEuclideanDistance(MahalanobisDistances):
    """
    computes the max-min normalized squared euclidean distance where

    .. math:: \\Sigma= I_{num_dims}
    """

    def __init__(self):
        super().__init__()
        self.num_dims_ = None
        self.max = None
        self.min = None

    def fit(self, num_dims: int, data_gen):
        """Fit Square Euclidean Distance metric

        Parameters
        -----------------
            num_dims: int
                the number of dimensions of the space in which the Squared Euclidean distance will be used.
        """

        self.num_dims_ = num_dims
        sigma = torch.eye(self.num_dims_).detach()
        self.max = data_gen.data_range[1]
        self.min = data_gen.data_range[0]
        super().fit(sigma)

    def __compute_dist__(self, X1, X2, sigma):
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
        dist = torch.sum((X_diff @ sigma) * X_diff, dim=-1, keepdim=True)
        return dist

    def adjust_length(self, X, Delta, goal):
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        if len(Delta.shape) == 1:
            Delta = Delta.unsqueeze(0)
        X2 = X + Delta

        cun_lenth = self.__compute_dist__(X, X2, self.sigma)
        scaling_factor = torch.sqrt(cun_lenth / goal)

        return scaling_factor*Delta - (scaling_factor-1)*self.min


class ProtectedSEDistances(MahalanobisDistances):
    '''Compute the Protected Squared Euclidean Distance metric which is similar to Squared Euclidean Distance
     while ignore the protected attributes while computing distance between data points.
    '''
    def __init__(self):
        super().__init__()
        self.num_dims_ = None
        self._protected_attributes = None

    def fit(self, protected_idx, num_dims:int):
        """Fit Causal Distance metric

        Parameters
        ------------
            protected_attributes: Iterable[int]
                List of attribute indices considered to be protected.
                The metric would ignore these attributes while
                computing distance between data points.
            num_dims: int
                Total number of attributes in the data points.
        """

        self.num_dims_ = num_dims
        self._protected_attributes = protected_idx
        sigma = torch.eye(self.num_dims_).detach()
        for p in self._protected_attributes:
            sigma[p][p] = 0.0
        super().fit(sigma)


class BinaryDistance(Distance):
    '''Compute the Binary Distance metric that is 0 when:
    .. math:: X1 = X2
    otherwise 1.
    '''
    def __init__(self):
        super().__init__()
        self._vectorized_dist = None

    @staticmethod
    def __compute_dist__(X1, X2):
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

        dist = torch.any(X1!=X2, dim=1).float()
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

                In the itemwise fashion (`itemwise_dist=False`), distance is
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

        if itemwise_dist:
            X1 = X1.unsqueeze(0) if len(X1.shape) == 0 else X1  # (D=1)
            X2 = X2.unsqueeze(0) if len(X2.shape) == 0 else X2  # (D=1)

            X1 = X1.unsqueeze(0).T if len(X1.shape) == 1 else X1  # (N, D)
            X2 = X2.unsqueeze(0).T if len(X2.shape) == 1 else X2  # (N, D)

            np.testing.assert_array_equal(
                X1.shape,
                X2.shape,
                err_msg="X1 and X2 should be of the same shape for itemwise distance computation",
            )
            dist = self.__compute_dist__(X1, X2)
        else:
            self.__init_vectorized_dist__()

            X1 = X1.unsqueeze(0).T if len(X1.shape) == 1 else X1  # (N, D)
            X2 = X2.unsqueeze(0).T if len(X2.shape) == 1 else X2  # (M, D)

            X1 = X1.unsqueeze(0) if len(X1.shape) == 2 else X1  # (B, N, D)
            X2 = X2.unsqueeze(0) if len(X2.shape) == 2 else X2  # (B, M, D)

            nsamples_x1 = X1.shape[1]
            nsamples_x2 = X2.shape[1]
            dist_shape = (-1, nsamples_x1, nsamples_x2)

            dist = self._vectorized_dist(X1, X2).view(dist_shape)

        return dist
    
class CausalDistance(BinaryDistance):
    def __init__(self):
        super().__init__()
        self.register_buffer('protected_vectors', torch.Tensor())
        self.num_dims = None
        self._protected_attributes = None
    
    def to(self, device):
        """Moves distance metric to a particular device

        Parameters
        ------------
            device: torch.device
        """

        assert (
            self.protected_vector is not None and len(self.protected_vector.size()) != 0
        ), "Please fit the metric before moving parameters to device"

        self.device = device
        self.protected_vector = self.protected_vector.to(self.device)

    def fit(self, protected_idx, num_dims):
        """Fit Causal Distance metric

        Parameters
        ------------
            protected_attributes: Iterable[int]
                List of attribute indices considered to be protected.
                The metric would ignore these protected attributes while
                computing distance between data points.
            num_dims: int
                Total number of attributes in the data points.
        """

        self._protected_attributes = protected_idx
        self.num_dims = num_dims

        self.protected_vector = torch.ones(num_dims)
        self.protected_vector[protected_idx] = 0.0

    def forward(self, X1, X2, itemwise_dist=True):
        """
        :param x, y: a B x D matrices
        :return: B x 1 matrix with the protected distance camputed between x and y
        """
        protected_X1 = (X1 * self.protected_vector)
        protected_X2 = (X2 * self.protected_vector)

        super().forward(protected_X1, protected_X2, itemwise_dist)
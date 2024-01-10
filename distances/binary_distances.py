import torch
import numpy as np
from torch import vmap
from distances.distance import Distance

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
        self.register_buffer('sensitive_vectors', torch.Tensor())
        self.num_dims = None
        self._sensitive_attributes = None
    
    def to(self, device):
        """Moves distance metric to a particular device

        Parameters
        ------------
            device: torch.device
        """

        assert (
            self.sensitive_vector is not None and len(self.sensitive_vector.size()) != 0
        ), "Please fit the metric before moving parameters to device"

        self.device = device
        self.sensitive_vector = self.sensitive_vector.to(self.device)

    def fit(self, sensitive_idx, num_dims):
        """Fit Causal Distance metric

        Parameters
        ------------
            sensitive_attributes: Iterable[int]
                List of attribute indices considered to be sensitive.
                The metric would ignore these sensitive attributes while
                computing distance between data points.
            num_dims: int
                Total number of attributes in the data points.
        """

        self._sensitive_attributes = sensitive_idx
        self.num_dims = num_dims

        self.sensitive_vector = torch.ones(num_dims)
        self.sensitive_vector[sensitive_idx] = 0.0

    def forward(self, X1, X2, itemwise_dist=True):
        """
        :param x, y: a B x D matrices
        :return: B x 1 matrix with the sensitive distance camputed between x and y
        """
        sensitive_X1 = (X1 * self.sensitive_vector)
        sensitive_X2 = (X2 * self.sensitive_vector)

        super().forward(sensitive_X1, sensitive_X2, itemwise_dist)
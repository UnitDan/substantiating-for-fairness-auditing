import torch
from distances.normalized_mahalanobis_distances import MahalanobisDistances
from typing import Iterable
from data.data_utils import DataGenerator
import numpy as np
from sklearn.linear_model import LogisticRegression

from inFairness.utils import datautils, validationutils

class SensitiveSubspaceDistance(MahalanobisDistances):
    """Implements Sensitive Subspace metric base class that accepts the
    basis vectors of a sensitive subspace, and computes a projection
    that ignores the sensitive subspace.

    The projection from the sensitive subspace basis vectors (A) is computed as:

    .. math:: P^{'} = I - (A (A A^{T})^{-1} A^{T})
    """

    def __init__(self):
        super().__init__()

    def fit(self, basis_vectors, data_gen: DataGenerator):
        """Fit Sensitive Subspace Distance metric

        Parameters
        --------------
            basis_vectors: torch.Tensor
                Basis vectors of the sensitive subspace
        """

        sigma = self.compute_projection_complement(basis_vectors)
        num_dims = sigma.shape[0]
        super().fit(num_dims, sigma, data_gen)

    def compute_projection_complement(self, basis_vectors):
        """Compute the projection complement of the space
        defined by the basis_vectors:

        projection complement given basis vectors (A) is computed as:

        .. math:: P^{'} = I - (A (A A^{T})^{-1} A^{T})

        Parameters
        -------------
            basis_vectors: torch.Tensor
                Basis vectors of the sensitive subspace
                Dimension (d, k) where d is the data features dimension
                and k is the number of sensitive dimensions

        Returns
        ----------
            projection complement: torch.Tensor
                Projection complement computed as described above.
                Shape (d, d) where d is the data feature dimension
        """

        # Computing the orthogonal projection
        # V(V V^T)^{-1} V^T
        projection = torch.linalg.inv(torch.matmul(basis_vectors.T, basis_vectors))

        projection = torch.matmul(basis_vectors, projection)

        # Shape: (n_features, n_features)
        projection = torch.matmul(projection, basis_vectors.T)

        # Complement the projection as: (I - Proj)
        projection_complement_ = torch.eye(projection.shape[0]) - projection
        projection_complement_ = projection_complement_.detach()

        return projection_complement_


class LogisticRegSensitiveSubspace(SensitiveSubspaceDistance):
    """Implements the Softmax Regression model based fair metric as defined in Appendix B.1
    of "Training individually fair ML models with sensitive subspace robustness" paper.

    This metric assumes that the sensitive attributes are discrete and observed for a small subset
    of training data. Assuming data of the form :math:`(X_i, K_i, Y_i)` where :math:`K_i` is the
    sensitive attribute of the i-th subject, the model fits a softmax regression model to the data as:

    .. math:: \mathbb{P}(K_i = l\\mid X_i) = \\frac{\exp(a_l^TX_i+b_l)}{\\sum_{l=1}^k \\exp(a_l^TX_i+b_l)},\\ l=1,\\ldots,k

    Using the span of the matrix :math:`A=[a_1, \cdots, a_k]`, the fair metric is trained as:

    .. math:: d_x(x_1,x_2)^2 = (x_1 - x_2)^T(I - P_{\\text{ran}(A)})(x_1 - x_2)

    References
    -------------
        `Yurochkin, Mikhail, Amanda Bower, and Yuekai Sun. "Training individually fair
        ML models with sensitive subspace robustness." arXiv preprint arXiv:1907.00020 (2019).`

    Additionally, we implement it to the max-min normalized version.
    """

    def __init__(self):
        super().__init__()
        self.basis_vectors_ = None
        self._logreg_models = None

    @property
    def logistic_regression_models(self):
        """Logistic Regression models trained by the metric to predict each sensitive attribute
        given inputs. The property is a list of logistic regression models each corresponding to
        :math:`\mathbb{P}(K_i = l\\mid X_i)`. This property can be used to measure the performance
        of the logistic regression models.
        """
        return self._logreg_models

    def fit(
        self,
        data_X: torch.Tensor,
        data_gen: DataGenerator,
        data_SensitiveAttrs: torch.Tensor = None,
        sensitive_idxs: Iterable[int] = None,
        keep_sensitive_idxs: bool = True,
        autoinfer_device: bool = True,
    ):
        """Fit Logistic Regression Sensitive Subspace distance metric

        Parameters
        --------------
            data_X: torch.Tensor
                Input data corresponding to either :math:`X_i` or :math:`(X_i, K_i)` in the equation above.
                If the variable corresponds to :math:`X_i`, then the `y_train` parameter should be specified.
                If the variable corresponds to :math:`(X_i, K_i)` then the `sensitive_idxs` parameter
                should be specified to indicate the sensitive attributes.

            data_SensitiveAttrs: torch.Tensor
                Represents the sensitive attributes ( :math:`K_i` ) and is used when the `X_train` parameter
                represents :math:`X_i` from the equation above. **Note**: This parameter is mutually exclusive
                with the `sensitive_idxs` parameter. Specififying both the `data_SensitiveAttrs` and `sensitive_idxs`
                parameters will raise an error

            sensitive_idxs: Iterable[int]
                If the `X_train` parameter above represents :math:`(X_i, K_i)`, then this parameter is used
                to provide the indices of sensitive attributes in `X_train`. **Note**: This parameter is mutually exclusive
                with the `sensitive_idxs` parameter. Specififying both the `data_SensitiveAttrs` and `sensitive_idxs`
                parameters will raise an error

            keep_sensitive_indices: bool
                True, if while training the model, sensitive attributes will be part of the training data
                Set to False, if for training the model, sensitive attributes will be excluded
                Default = True

            autoinfer_device: bool
                Should the distance metric be automatically moved to an appropriate
                device (CPU / GPU) or not? If set to True, it moves the metric
                to the same device `X_train` is on. If set to False, keeps the metric
                on CPU.
        """

        if data_SensitiveAttrs is not None and sensitive_idxs is None:
            data_range = data_gen.get_range('data', include_sensitive_feature=False)
            data_X = (data_X - data_range[0])/(data_range[1] - data_range[0])

            basis_vectors_ = self.compute_basis_vectors_data(
                X_train=data_X, y_train=data_SensitiveAttrs
            )

        elif data_SensitiveAttrs is None and sensitive_idxs is not None:
            data_range = data_gen.get_range('data', include_sensitive_feature=True)
            data_X = (data_X - data_range[0])/(data_range[1] - data_range[0])

            basis_vectors_ = self.compute_basis_vectors_sensitive_idxs(
                data_X,
                sensitive_idxs=sensitive_idxs,
                keep_sensitive_idxs=keep_sensitive_idxs,
            )

        else:
            raise AssertionError(
                "Parameters `y_train` and `sensitive_idxs` are exclusive. Either of these two parameters should be None, and cannot be set to non-None values simultaneously."
            )

        super().fit(basis_vectors_, data_gen)
        self.basis_vectors_ = basis_vectors_

        if autoinfer_device:
            device = datautils.get_device(data_X)
            super().to(device)

    def compute_basis_vectors_sensitive_idxs(
        self, data, sensitive_idxs, keep_sensitive_idxs=True
    ):

        dtype = data.dtype

        data = datautils.convert_tensor_to_numpy(data)
        basis_vectors_ = []
        num_attr = data.shape[1]

        # Get input data excluding the sensitive attributes
        sensitive_idxs = sorted(sensitive_idxs)
        free_idxs = [idx for idx in range(num_attr) if idx not in sensitive_idxs]
        X_train = data[:, free_idxs]
        Y_train = data[:, sensitive_idxs]

        self.__assert_sensitiveattrs_binary__(Y_train)

        self._logreg_models = [
            LogisticRegression(solver="liblinear", penalty="l1")
            .fit(X_train, Y_train[:, idx])
            for idx in range(len(sensitive_idxs))
        ]

        coefs = np.array(
            [
                self._logreg_models[idx].coef_.squeeze()
                for idx in range(len(sensitive_idxs))
            ]
        )  # ( |sensitive_idxs|, |free_idxs| )

        if keep_sensitive_idxs:
            # To keep sensitive indices, we add two basis vectors
            # First, with logistic regression coefficients with 0 in
            # sensitive indices. Second, with one-hot vectors with 1 in
            # sensitive indices.

            basis_vectors_ = np.empty(shape=(2 * len(sensitive_idxs), num_attr))

            for i, sensitive_idx in enumerate(sensitive_idxs):

                sensitive_basis_vector = np.zeros(shape=(num_attr))
                sensitive_basis_vector[sensitive_idx] = 1.0

                unsensitive_basis_vector = np.zeros(shape=(num_attr))
                np.put_along_axis(
                    unsensitive_basis_vector, np.array(free_idxs), coefs[i], axis=0
                )

                basis_vectors_[2 * i] = unsensitive_basis_vector
                basis_vectors_[2 * i + 1] = sensitive_basis_vector
        else:
            # Protected indices are to be discarded. Therefore, we can
            # simply return back the logistic regression coefficients
            basis_vectors_ = coefs

        basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
        basis_vectors_ = basis_vectors_.detach()

        return basis_vectors_

    def compute_basis_vectors_data(self, X_train, y_train):

        dtype = X_train.dtype

        X_train = datautils.convert_tensor_to_numpy(X_train)
        y_train = datautils.convert_tensor_to_numpy(y_train)

        self.__assert_sensitiveattrs_binary__(y_train)

        basis_vectors_ = []
        outdim = y_train.shape[-1]

        self._logreg_models = [
            LogisticRegression(solver="liblinear", penalty="l1")
            .fit(X_train, y_train[:, idx])
            for idx in range(outdim)
        ]

        basis_vectors_ = np.array(
            [
                self._logreg_models[idx].coef_.squeeze()
                for idx in range(outdim)
            ]
        )

        basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
        basis_vectors_ = basis_vectors_.detach()

        return basis_vectors_

    def __assert_sensitiveattrs_binary__(self, sensitive_attrs):

        assert validationutils.is_tensor_binary(
            sensitive_attrs
        ), "Sensitive attributes are required to be binary to learn the metric. Please binarize these attributes before fitting the metric."
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

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

from abc import ABC, abstractmethod
from typing import Union
import gpytorch
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module


class GPytorchModel(ABC, Module):
    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def posterior(self,
                  X: Tensor,
                  apply_likelihood: Union[bool, Tensor] = False,
                  *args, **kwargs) -> Union[MultivariateNormal, Categorical]:
        pass

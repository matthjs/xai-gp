from abc import ABC, abstractmethod
from typing import Union
import gpytorch
from torch import Tensor
from torch.nn import Module


class GPytorchModel(ABC, Module):
    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def posterior(self,
                  X: Tensor,
                  observation_noise: Union[bool, Tensor] = False,
                  *args, **kwargs) -> gpytorch.distributions.MultivariateNormal:
        pass

import json
from typing import Any, Dict, List, Union
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, SoftmaxLikelihood
from gpytorch.models.deep_gps.dspp import DSPP
from pandas import Categorical
from torch import Tensor
from xai_gp.models.gp.deepgplayers import DSPPHiddenLayer
from xai_gp.models.gp.gpbase import GPytorchModel


class DSPPModel(DSPP, GPytorchModel):

    def __init__(self,
                 input_dim: int,    # train_x_shape[-1]
                 hidden_layers_config: List[Dict[str, Any]],
                 Q: int = 8,
                 num_inducing_points: int = 128,
                 input_transform: Any = None,
                 outcome_transform: Any = None,
                 classification: bool = False,
                 **kwargs):
        super().__init__(num_quad_sites=Q)
        input_dims = input_dim
        self.layers = []
        self.Q = Q

        hidden_layers_config = json.loads(hidden_layers_config) if isinstance(hidden_layers_config, str) else hidden_layers_config

        # Build hidden layers
        for layer_config in hidden_layers_config:
            hidden_layer = DSPPHiddenLayer(
                input_dims=input_dims,
                output_dims=layer_config['output_dims'],
                num_inducing=num_inducing_points,
                mean_type=layer_config['mean_type'],
                Q=Q
            )
            self.layers.append(hidden_layer)
            input_dims = hidden_layer.output_dims if hidden_layer.output_dims else 1

        self.out_dim = input_dims
        self.layers = torch.nn.ModuleList(self.layers)
        self.likelihood = SoftmaxLikelihood(num_classes=100, num_features=8) if classification else GaussianLikelihood()
        self._num_outputs = 1
        self.double()
        self.intermediate_outputs = None

        # Initialize transforms
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform

    def forward(self, inputs: Tensor, **kwargs) -> MultivariateNormal:
        x = inputs
        self.intermediate_outputs = []
        for layer in self.layers:
            x = layer(x, **kwargs)
            self.intermediate_outputs.append(x)
        return x

    def posterior(
            self,
            X: Tensor,
            apply_likelihood: bool = False,  # Renamed from observation_noise
            *args, **kwargs
    ) -> Union[MultivariateNormal, Categorical]:
        """
        Compute the posterior distribution.

        :param X: Input tensor.
        :param apply_likelihood: Whether to apply the likelihood transformation.
                                 For classification, this returns class probabilities.
                                 For regression, this adds observation noise.
        :return: Posterior distribution.
        """
        self.eval()
        if self.input_transform is not None:
            X = self.input_transform(X)

        with torch.no_grad():
            dist = self(X, mean_input=X)
            if apply_likelihood:
                dist = self.likelihood(dist)

        if self.outcome_transform is not None:
            dist = self.outcome_transform.untransform(dist)
        return dist

    def get_intermediate_outputs(self) -> List[Tensor]:
        return self.intermediate_outputs

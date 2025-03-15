"""
Based on https://github.com/pytorch/botorch/issues/1750 and
https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html
"""
from typing import Any, Dict, List, Union, Tuple
import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from torch import Tensor
from xai_gp.models.deepgplayers import DeepGPHiddenLayer
from xai_gp.models.gpbase import GPytorchModel


class DeepGPModel(DeepGP, GPytorchModel):
    """
    Deep Gaussian Process Model class implementing BoTorch and GPyTorch interfaces.
    This class currently does not allow customization of kernel functions and uses RBF for each unit by default.
    """

    def __init__(self,
                 train_x_shape: torch.Size,
                 hidden_layers_config: List[Dict[str, Any]],
                 num_inducing_points: int = 128,
                 input_transform: Any = None,
                 outcome_transform: Any = None):
        """
        Constructor for DeepGPModel.

        :param train_x_shape: Shape of the training data.
        :param hidden_layers_config: List of dictionaries where each dictionary contains the configuration
                                     for a hidden layer. Each dictionary should have the keys:
                                     - "output_dims": Number of output dimensions.
                                     - "mean_type": Type of mean function ("linear" or "constant").
                                     NOTE: The last layer should always have mean_type as "constant".
        :param num_inducing_points: Number of inducing points (per unit) for the variational strategy. Default is 128.
        :param input_transform: Transformation to be applied to the inputs. Default is None.
        :param outcome_transform: Transformation to be applied to the outputs. Default is None.
        """
        super().__init__()

        input_dims = train_x_shape[-1]
        self.layers = []

        # Create hidden layers based on the provided configuration
        for layer_config in hidden_layers_config:
            hidden_layer = DeepGPHiddenLayer(
                input_dims=input_dims,
                output_dims=layer_config['output_dims'],
                mean_type=layer_config['mean_type'],
                num_inducing=num_inducing_points
            )
            self.layers.append(hidden_layer)
            input_dims = layer_config['output_dims']

        # Add all layers as module list
        self.layers = torch.nn.ModuleList(self.layers)
        self.likelihood = GaussianLikelihood()   # For classification I believe replace with SoftMaxLikelihood.
        self._num_outputs = 1
        self.double()
        self.intermediate_outputs = None

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the model.
        Side effect: stores intermediate output representations in a list.

        :param inputs: Input tensor.
        :return: Output tensor after passing through the hidden layers.
        """
        x = inputs
        self.intermediate_outputs = []
        for layer in self.layers:
            x = layer(x)
            self.intermediate_outputs.append(x)
        return x

    def posterior(self,
                  X: Tensor,
                  observation_noise: Union[bool, Tensor] = False,
                  *args, **kwargs) -> gpytorch.distributions.MultivariateNormal:
        """
        Compute the posterior distribution.

        :param X: Input tensor.
        :param observation_noise: Whether to add observation noise. Default is False.
        :return: Posterior distribution.
        """
        self.eval()  # make sure model is in eval mode

        X = self.transform_inputs(X)  # Transform the inputs

        with torch.no_grad() and gpytorch.settings.num_likelihood_samples(10):
            dist = self(X)  # Compute the posterior distribution

            if observation_noise:
                dist = self.likelihood(dist, *args, **kwargs)  # Add observation noise

        if self.outcome_transform is not None:
            # Ensure you have a GPyTorch-compatible transform here
            dist = self.outcome_transform.untransform(dist)
        return dist

    def get_intermediate_outputs(self) -> List[Tensor]:
        """
        Get the intermediate outputs from the hidden layers.
        Prerequisite: A forward pass must have been performed.

        :return: List of intermediate outputs.
        """
        return self.intermediate_outputs

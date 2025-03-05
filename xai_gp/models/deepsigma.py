from typing import Any, Dict, List, Union, Tuple
import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from torch import Tensor
from torch.utils.data import DataLoader
from xai_gp.models.deepgplayers import DSPPHiddenLayer
from xai_gp.models.gpbase import GPytorchModel


class DSPPModel(DSPP, GPytorchModel):

    def __init__(self,
                 train_x_shape: torch.Size,
                 hidden_layers_config: List[Dict[str, Any]],
                 Q: int = 8,
                 num_inducing_points: int = 128,
                 input_transform: Any = None,
                 outcome_transform: Any = None):
        super().__init__(Q=Q)

        input_dims = train_x_shape[-1]
        self.layers = torch.nn.ModuleList()
        self.Q = Q

        # Build hidden layers
        for layer_config in hidden_layers_config:
            layer = DSPPHiddenLayer(
                input_dims=input_dims,
                output_dims=layer_config['output_dims'],
                num_inducing=num_inducing_points,
                mean_type=layer_config['mean_type'],
                Q=Q
            )
            self.layers.append(layer)
            input_dims = layer.output_dims if layer.output_dims else 1

        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
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

    def posterior(self,
                  X: Tensor,
                  observation_noise: Union[bool, Tensor] = False,
                  *args, **kwargs) -> MultivariateNormal:
        self.eval()
        X = self.transform_inputs(X)

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
            dist = self(X, mean_input=X)
            if observation_noise:
                dist = self.likelihood(dist)

        if self.outcome_transform is not None:
            dist = self.outcome_transform.untransform(dist)
        return dist

    def predict(self, test_loader: DataLoader) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, solves=False):
            mus, variances, lls = [], [], []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch, mean_input=x_batch))

                # Compute log likelihood using quadrature weights
                base_ll = self.likelihood.log_marginal(y_batch, self(x_batch))
                weighted_ll = self.quad_weights.unsqueeze(-1) + base_ll
                batch_ll = weighted_ll.logsumexp(dim=0)

                # Compute weighted mean and variance
                weights = torch.softmax(self.quad_weights, dim=0)
                comp_means = preds.mean
                comp_vars = preds.variance

                overall_mean = (weights.unsqueeze(-1) * comp_means).sum(0)
                diff = comp_means - overall_mean.unsqueeze(0)
                overall_var = (weights.unsqueeze(-1) * (comp_vars + diff.pow(2))).sum(0)

                mus.append(overall_mean)
                variances.append(overall_var)
                lls.append(batch_ll)

        return (
            torch.cat(mus, dim=0),
            torch.cat(variances, dim=0),
            torch.cat(lls, dim=0),
        )

    def get_intermediate_outputs(self) -> List[Tensor]:
        return self.intermediate_outputs

from typing import Callable

import torch
from torch import nn
from xai_gp.models.ensemble.ensemblebase import DeepEnsemble


class DeepEnsembleRegressor(DeepEnsemble):
    """
    Implementation of ensemble regressor for UQ.
    Use nll_loss for training.
    """

    def __init__(self, model_fn: Callable, num_models: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])

    def forward(self, x: torch.Tensor, disentangle_uncertainty: bool = False) -> tuple:
        """
        Output
        p(y | x) ∼ N(μ∗(x), σ2∗(x))
        :param x: input tensor
        :param disentangle_uncertainty
        Returns: Gaussian Mixture with mean and variance.
        """
        means, variances = zip(*[model(x) for model in self.models])

        means = torch.stack(means)  # Shape: (M, batch, output_dim)
        variances = torch.stack(variances)  # Shape: (M, batch, output_dim)

        mean_ensemble = means.mean(dim=0)
        var_ensemble = (variances + torch.square(means)).mean(dim=0) - mean_ensemble

        if disentangle_uncertainty:
            ale_var = variances.mean(dim=0)
            epi_var = means.var(dim=0)
            return mean_ensemble, var_ensemble, ale_var, epi_var

        return mean_ensemble, var_ensemble






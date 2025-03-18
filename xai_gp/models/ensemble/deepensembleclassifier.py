from typing import Callable, Union
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from xai_gp.models.ensemble.ensemblebase import DeepEnsemble


def sampling_softmax(means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    """
    Given a set of means and variances, sample from the Gaussian and pass the samples to the
    softmax to get class probabilities.
    :param means
    :param variances
    :returns: class probabilities (torch.Tensor).
    """
    # Sample from each associated MVN.
    samples = torch.stack([MultivariateNormal(mean, torch.diag_embed(variance)).rsample() \
                           for mean, variance in zip(means, variances)])
    # Sampling softmax
    prob = F.softmax(samples).mean(dim=0)  # p(y|x)

    return prob


class DeepEnsembleClassifier(DeepEnsemble):
    """
    Implementation of ensemble regressor for UQ.
    Use nll_loss for training.
    """

    def __init__(self, model_fn: Callable, num_models: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])

    def forward(self, x: torch.Tensor, disentangle_uncertainty: bool = False) -> Union[torch.Tensor, tuple]:
        """
        Output
        p(y | x) ∼ N(μ∗(x), σ2∗(x))
        :param x: input tensor
        :param disentangle_uncertainty
        Returns: Gaussian Mixture with mean and variance.
        """
        means, variances = zip(*[model(x) for model in self.models])

        prob = sampling_softmax(means, variances)
        means = torch.stack(means)  # Shape: (M, batch, output_dim)
        variances = torch.stack(variances)  # Shape: (M, batch, output_dim)

        if disentangle_uncertainty:
            ale_var = variances.mean(dim=0)
            epi_var = means.var(dim=0)

            prob_ale = sampling_softmax(means, ale_var)
            prob_epi = sampling_softmax(means, epi_var)

            return prob, prob_ale, prob_epi

        return prob

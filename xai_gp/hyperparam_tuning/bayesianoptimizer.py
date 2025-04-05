from typing import Dict, Optional, Callable, Tuple, List
import torch
from ax import Experiment
from ax.service.ax_client import AxClient
import torch
from typing import Dict, Any, Optional
from ax.service.utils.instantiation import ObjectiveProperties


class BayesianOptimizer:
    def __init__(
            self,
            search_space: List[dict],
            model_factory: Callable[[Dict[str, Any]], torch.nn.Module],
            train_fn: Callable[[torch.nn.Module, Dict[str, Any]], float],
            eval_fn: Callable[[torch.nn.Module], Dict[str, float]],
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            tracking_metrics: Optional[Tuple[str]] = None,
            objective_name: str = "loss",
            minimize: bool = True,
    ):
        """
        Generic Bayesian optimization wrapper for various models.

        :param search_space: Ax parameter search space definition
        :param model_factory: Function that creates model from parameters
            A function that takes in a dictionary of hyperparameters and returns a nn.Module instance.
        :param train_fn: Function that trains model and returns loss
            The train function expects a model (nn.Module) and a dictionary of parameters
            and returns a float value (e.g., the training loss).
        :param eval_fn: Function that evaluates model and returns metrics
            The evaluation function expects a model (nn.Module) and returns a dictionary of metrics.
        :param device: Target device for computation
        :param tracking_metrics: Metrics to track (defaults to eval_fn keys)

        Bayesian optimization loop:
        1: Initialize with a set of random hyperparameters and evaluate objective f(x) at these points.
        2: Fit the surrogate model (e.g., Gaussian process) to the observed data {x_i, f(x_i)}.
        3: A promising new set of hyperparameters x_new are queried that maximize the acquisition function
        (a lot of them involve the uncertainty variance of the GP predictive posterior distribution)
        4: Evaluate f(x_new)
        5: Augment dataset with {(x_new, f(x_new)}.
        6: Goto 1. until stopping criterion is met.
        """
        self.ax_client = AxClient(torch_device=device)
        self.device = device
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.tracking_metrics = tracking_metrics

        self.objective_name = objective_name
        self.minimize = minimize

        # Initialize hyperparameter tuning experiment. We want to find the optimal set of
        # hyperparameters such that an objective is minimized (or maximized)
        self.ax_client.create_experiment(
            name="bayesian_optimization",
            parameters=search_space,
            objectives={self.objective_name: ObjectiveProperties(minimize=self.minimize)},
        )

    def run_trial(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single optimization trial"""
        try:
            model = self.model_factory(params).to(self.device)

            train_loss = self.train_fn(model, params)

            metrics = self.eval_fn(model)
            metrics["train_loss"] = train_loss  # Always track training loss

            return metrics

        except Exception as e:
            print(f"Trial failed with params {params}: {str(e)}")
            return {m: float("inf") for m in self.tracking_metrics}

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """Main optimization loop"""
        for _ in range(n_trials):
            # Get suggested hyperparameter values.
            params, trial_idx = self.ax_client.get_next_trial()
            metrics = self.run_trial(params)

            # Report primary metric (loss) to Ax
            self.ax_client.complete_trial(
                trial_index=trial_idx,
                raw_data=metrics[self.objective_name]
            )

        return self.get_best_parameters()

    def get_best_parameters(self) -> Dict[str, Any]:
        """Return best parameters found"""
        return self.ax_client.get_best_parameters()[0]

    @property
    def experiment(self) -> Experiment:
        """Access underlying Ax experiment object."""
        return self.ax_client.experiment

    def save_state(self, filename: str) -> None:
        """Save experiment state to file."""
        self.ax_client.save_to_json_file(filename)

    @classmethod
    def load_state(cls, filename: str, train_data: Dict[str, torch.Tensor]) -> "BayesianOptimizer":
        """Load existing experiment from file."""
        new_instance = cls.__new__(cls)
        new_instance.ax_client = AxClient.load_from_json_file(filename)
        new_instance.train_data = train_data
        return new_instance

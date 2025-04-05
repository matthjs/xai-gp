import json

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from functools import partial
from typing import Dict, Any

from xai_gp.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from xai_gp.hyperparam_tuning.wrappers import generic_model_factory, train_gp, evaluate_model, print_metrics
from xai_gp.models.gp import DeepGPModel


def test_bayesian_optimizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.rand(100, 5).to(device=device)  # 100 samples, 5 features
    y = X @ torch.randn(5, 1).to(device=device) + 0.1 * torch.randn(100, 1).to(device=device)  # Linear relationship with noise
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=10)
    test_loader = DataLoader(dataset, batch_size=10)  # Using same data for simplicity

    # Not sure if this is the best approach for differentiating between fixed
    # hyperparameters and variable hyperparameters but this will do.
    search_space = [
        {
            "name": "input_dim",
            "type": "fixed",
            "value": X.shape[-1],
        },
        {
            "name": "hidden_layers_config",
            "type": "choice",
            "values": [
                json.dumps([{"output_dims": None, "mean_type": "constant"}]),
                json.dumps([
                    {"output_dims": 4, "mean_type": "linear"},
                    {"output_dims": None, "mean_type": "constant"}
                ]),
                json.dumps([
                    {"output_dims": 8, "mean_type": "linear"},
                    {"output_dims": 4, "mean_type": "linear"},
                    {"output_dims": None, "mean_type": "constant"}
                ]),
                json.dumps([
                    {"output_dims": 16, "mean_type": "linear"},
                    {"output_dims": None, "mean_type": "constant"}
                ])
            ]
        },
        {"name": "num_inducing", "type": "range", "bounds": [64, 256]},
        {"name": "lr", "type": "range", "bounds": [1e-4, 0.1], "log_scale": True},
        {"name": "num_epochs", "type": "range", "bounds": [5, 10]}
    ]

    # Can also use generic model factory but I think I need to adjust that a bit.
    model_factory = partial(
        generic_model_factory,
        model_type='DGP')

    eval_fn = partial(evaluate_model,
                      test_loader=test_loader,
                      task_type="regression")

    bayesian_optimizer = BayesianOptimizer(
        search_space=search_space,
        model_factory=model_factory,
        train_fn=partial(train_gp,
                         data_loader=train_loader,
                         gp_mode='DGP'),
        eval_fn=eval_fn,
        objective_name='rmse',  # Must correspond to one of the keys in the dict returned by the
        # eval function
        minimize=True,
        # tracking_metrics=("mae", "r2", "calibration_error")   # Also track these
    )

    # 7. Run optimization with reduced trials for testing
    best_params = bayesian_optimizer.optimize(n_trials=5)

    # 8. Validate results
    assert isinstance(best_params, dict), "Should return dictionary of best parameters"
    required_keys = {"num_layers", "hidden_size", "num_inducing", "r", "num_epochs"}
    assert all(k in best_params for k in required_keys), "Missing required parameters"

    # 9. Test final model with best parameters
    best_model = model_factory(best_params)
    final_metrics = eval_fn(best_model)

    print("\nOptimization Results:")
    print_metrics(final_metrics)
    print("Best Parameters:", best_params)

    # Basic sanity checks
    assert final_metrics['rmse'] < 10.0, "RMSE unreasonably high"
    assert final_metrics['r2'] > 0.5, "R² score too low"
    assert final_metrics['calibration_error'] < 0.2, "Poor calibration"

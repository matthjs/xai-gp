"""
Utility functions for the BayesianOptimizer.
"""
from typing import Dict, Any
from torch import nn, optim
from torch.utils.data import DataLoader
from xai_gp.models.ensemble import DeepEnsembleRegressor, DeepEnsembleClassifier, train_ensemble_regression, \
    train_ensemble_classification
from xai_gp.models.gp import DSPPModel, DeepGPModel, fit_gp


def generic_model_factory(params: Dict[str, Any], model_type: str = 'DeepGP') -> nn.Module:
    """
    Factory function for models
    :param params: Dictionary of arguments passed to the constructor
    :param model_type: 'DGP' or 'DSPP', 'DeepEnsembleRegressor', 'DeepEnsembleClassifier'

    Tip: Use partial when passing this as a callable ->
    partial(create_gp, gp_mode='DGP') for a DGP factory function.
    partial(create_gp, gp_mode='DSPP') for a DSPP factory function.
    """
    if model_type == 'DSPPModel':
        return DSPPModel(**params)
    elif model_type == 'DeepGPModel':
        return DeepGPModel(**params)
    elif model_type == 'DeepEnsembleRegressor':
        return DeepEnsembleRegressor(**params)
    elif model_type == 'DeepEnsembleClassifier':
        return DeepEnsembleClassifier(**params)

    print("model_type", model_type)
    raise ValueError('Invalid gp_mode')


def train_gp(model: nn.Module, params: Dict[str, Any], data_loader: DataLoader, gp_mode: str) -> float:
    """
    Use partial when passing to Bayesian optimizer
    partial(train_gp, data_loader=some_data_loader, gp_mode='DGP')
    :param model: model
    :param params: arguments
    :param data_loader
    :param gp_mode
    """
    # For simplicity fix the optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=params['lr'])  # Modify this if you want more hyperparams to vary.

    return fit_gp(
        model=model,  # Assume nn.Module is a GPytorch
        data_loader=data_loader,
        gp_mode=gp_mode,
        num_epochs=params['num_epochs'],
        optimizer=optimizer,
    )  # Returns train loss


def train_ensemble(model: nn.Module, params: Dict[str, Any], data_loader: DataLoader, task_type: str) -> float:
    """
    Use partial when passing to Bayesian optimizer
    """
    # Assumes fixed optimizer (Adam) and loss
    if task_type == "regression":
        return train_ensemble_regression(
            model, data_loader,
            num_epochs=params['num_epochs'],
            lr=params['lr']
        )
    elif task_type == "classification":
        return train_ensemble_classification(
            model, data_loader,
            num_epochs=params['num_epochs'],
            lr=params['lr']
        )
    else:
        raise ValueError("Invalid task type.")


def print_metrics(metrics: dict, task_type: str = "regression") -> None:
    """Print formatted evaluation metrics."""
    print("\nEvaluation Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")

    if task_type == "regression":
        print(f"Calibration Error: {metrics['calibration_error']:.4f}")
        print(f"Sharpness (Avg Variance): {metrics['sharpness']:.4f}")

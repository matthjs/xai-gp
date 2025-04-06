from typing import Dict, Any
from torch import nn, optim
from torch.utils.data import DataLoader
from xai_gp.models.ensemble import DeepEnsembleRegressor, DeepEnsembleClassifier, train_ensemble_regression, \
    train_ensemble_classification
from xai_gp.models.gp import DSPPModel, DeepGPModel, fit_gp
import numpy as np
import torch
from sklearn.metrics import r2_score

from xai_gp.utils.calibration import regressor_calibration_error, regressor_calibration_curve
from xai_gp.utils.evaluation import extract_predictions, is_gp_model


def generic_model_factory(params: Dict[str, Any], model_type: str = 'DeepGP') -> nn.Module:
    """
    Factory function for models
    :param params: Dictionary of arguments passed to the constructor
    :param model_type: 'DGP' or 'DSPP', 'DeepEnsembleRegressor', 'DeepEnsembleClassifier'

    Tip: Use partial when passing this as a callable ->
    partial(create_gp(gp_mode='DGP')) for a DGP factory function.
    partial(create_gp(gp_mode='DSPP')) for a DSPP factory function.
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
    partial(train_gp(data_loader=some_data_loader, gp_mode='DGP'))
    :param model: model
    :param params: arguments
    :param data_loader
    :param gp_mode
    """
    # For simplicity I will fixed the optimizer here
    optimizer = optim.Adam(model.parameters(),
                           lr=params['lr'])  # Modify this if you want more hyperparams to vary.

    return fit_gp(
        model=model,  # Assume nn.Module is a GPytorch
        data_loader=data_loader,
        gp_mode=gp_mode,
        num_epochs=params['num_epochs'],
        optimizer=optimizer,
        # Beta by default is 0.05
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


def evaluate_model(model: nn.Module, test_loader, task_type="regression") -> Dict[str, Any]:
    """
    Evaluate model performance on test data.
    Code duplication w.r.t. what is in main but I need it seperately.
    In BayesianOptimizer use partial! e.g.
    partial(evaluate_model(test_loader=test_loader, task_type="regression")

    Args:
        model: Trained model (GP or ensemble)
        test_loader: DataLoader containing test data
        task_type: Type of task ("regression" or "classification")

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_means = []
    all_variances = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            means, variances = extract_predictions(model, batch_x)

            all_means.append(means)
            all_variances.append(variances)
            all_targets.append(batch_y)

        # Concatenate all predictions and targets
        if is_gp_model(model):
            if len(all_means[0].shape) == 1:
                test_means = torch.cat(all_means, dim=0).cpu()
                test_variances = torch.cat(all_variances, dim=0).cpu()
            else:
                test_means = torch.cat(all_means, dim=1).cpu()
                test_variances = torch.cat(all_variances, dim=1).cpu()
        else:
            test_means = torch.cat(all_means, dim=0).cpu()
            test_variances = torch.cat(all_variances, dim=0).cpu()

        test_targets = torch.cat(all_targets, dim=0).cpu()

    # Calculate basic regression metrics
    metrics = {
        'mae': torch.mean(torch.abs(test_means - test_targets)).item(),
        'rmse': torch.sqrt(torch.mean((test_means - test_targets) ** 2)).item(),
        'r2': r2_score(test_targets.numpy(), test_means.numpy()),
    }

    # Calculate uncertainty metrics if regression
    if task_type == "regression":
        test_stds = torch.sqrt(test_variances).numpy()
        test_means = test_means.numpy()
        test_targets = test_targets.numpy()

        # Calibration metrics
        calibration_error = regressor_calibration_error(test_means, test_targets, test_stds)
        conf, acc = regressor_calibration_curve(test_means, test_targets, test_stds)

        metrics.update({
            'calibration_error': calibration_error,
            'calibration_curve': (conf, acc),
            'sharpness': np.mean(test_stds ** 2),  # Average variance
        })
    else:
        raise NotImplementedError("Classification metrics not implemented!!!")

    return metrics


def print_metrics(metrics, task_type="regression"):
    """Print formatted evaluation metrics."""
    print("\nEvaluation Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")

    if task_type == "regression":
        print(f"Calibration Error: {metrics['calibration_error']:.4f}")
        print(f"Sharpness (Avg Variance): {metrics['sharpness']:.4f}")

import json
import os
from functools import partial
from typing import Dict, Any, Tuple
from omegaconf import OmegaConf

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from xai_gp.hyperparam_tuning.bayesianoptimizer import BayesianOptimizer
from xai_gp.hyperparam_tuning.wrappers import (
    generic_model_factory, train_gp, train_ensemble, 
)
from xai_gp.utils.evaluation import is_gp_model
from xai_gp.models.ensemble import TwoHeadMLP
from xai_gp.utils.evaluation import evaluate_model


def load_search_space(search_space_path: str, input_dim: int) -> list:
    """
    Load and prepare the search space configuration for Bayesian optimization.
    
    Args:
        search_space_path: Path to the JSON file containing search space definition
        input_dim: Input dimension for the model
        
    Returns:
        List of parameter specifications for Ax
    """
    with open(search_space_path, 'r') as f:
        search_space = json.load(f)
    
    # Add fixed parameter for input dimension
    search_space.append({
        "name": "input_dim",
        "type": "fixed",
        "value": input_dim
    })
    
    return search_space


def create_model_instance(params: Dict[str, Any], model_type: str, cfg) -> torch.nn.Module:
    """
    Create a model instance based on parameters and model type.
    
    Args:
        params: Model parameters from the optimizer
        model_type: Type of model to create
        
    Returns:
        Instantiated model
    """
    
    # Handle parameters that need conversion
    if "hidden_layers_config" in params and isinstance(params["hidden_layers_config"], str):
        params["hidden_layers_config"] = json.loads(params["hidden_layers_config"])
    
    # Map parameter names to the ones expected by the models
    if "num_inducing_points" in params and model_type in ["DeepGPModel", "DSPPModel"]:
        params["num_inducing_points"] = params.pop("num_inducing_points")
        params["classification"] = cfg.data.task_type == "classification"
    
    # For Deep Ensemble models, create a base model function
    if model_type in ["DeepEnsembleRegressor", "DeepEnsembleClassifier"]:
        # Extract parameters for the base model only
        hidden_layers_config = params.get("hidden_layers_config", None)
        input_dim = params.pop("input_dim", None)
        
        # Create base model function
        base_model = lambda: TwoHeadMLP(
            input_dim=input_dim,
            output_dim=1,  # Handled by dataset
            hidden_layers_config=hidden_layers_config
        )
        
        # Set parameters for the ensemble model
        params["model_fn"] = base_model
        params["num_models"] = params.pop("num_ensemble_models", 5)  # Rename parameter
    
    # Use the generic model factory to create the model
    return generic_model_factory(params, model_type=model_type)


def run_hyperparameter_optimization(
    cfg: DictConfig, 
    train_loader: DataLoader, 
    test_loader: DataLoader,
    input_shape: Tuple[int, ...], 
    device: torch.device
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization using Bayesian optimization.
    
    Args:
        cfg: Configuration object
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        input_shape: Shape of input data
        device: Device to run on
        
    Returns:
        Dictionary of best parameters
    """
    model_type = cfg.model.type
    
    # Get base path for hyperparameter configs
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "conf")
    
    # Determine search space path based on model type
    if model_type == "DSPPModel":
        search_space_path = os.path.join(base_path, "hyperparam_dspp.json")
    elif model_type == "DeepGPModel":
        search_space_path = os.path.join(base_path, "hyperparam_deepgp.json") 
    elif model_type in ["DeepEnsembleRegressor", "DeepEnsembleClassifier"]:
        search_space_path = os.path.join(base_path, "hyperparam_ensemble_mlp.json")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Using search space configuration from: {search_space_path}")
    
    # Load search space and add input dimension
    search_space = load_search_space(search_space_path, input_shape[1])
    
    # Create model factory function
    model_factory = partial(create_model_instance, model_type=model_type, cfg=cfg)
    
    # Create training function
    if is_gp_model(model_type):
        gp_mode = cfg.model.gp_mode
        train_fn = partial(train_gp, data_loader=train_loader, gp_mode=gp_mode)
    else:
        task_type = cfg.data.task_type
        train_fn = partial(train_ensemble, data_loader=train_loader, task_type=task_type)
    
    # Create evaluation function
    eval_fn = partial(evaluate_model, test_loader=test_loader, cfg=cfg)
    
    objective = 'nll' if cfg.data.task_type == 'regression' else 'accuracy'
    minimize = True if objective in ['mae', 'nll', 'mse', 'calibration_error'] else False
    
    # Create and run optimizer
    optimizer = BayesianOptimizer(
        search_space=search_space,
        model_factory=model_factory,
        train_fn=train_fn,
        eval_fn=eval_fn,
        device=device,
        objective_name=objective,
        minimize=minimize,
    )
    
    print(f"Running Bayesian optimization with {cfg.hyperparam_tuning.n_trials} trials...")
    best_params = optimizer.optimize(n_trials=cfg.hyperparam_tuning.n_trials)
    
    # Add objective to best parameters
    best_params["objective"] = objective
    best_params["is_minimize"] = minimize
    
    # Print and return best parameters
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params


def get_best_model(best_params: Dict[str, Any], cfg: DictConfig, device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Create a model from the best parameters found by the optimizer.
    
    Args:
        best_params: Best parameters found by optimizer
        cfg: Configuration object
        device: Device to run on
        
    Returns:
        Tuple of (model, optimizer)
    """
    model_type = cfg.model.type
    model = create_model_instance(best_params, model_type, cfg).to(device)
    
    # Create optimizer
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(),
        lr=best_params.get("lr", cfg.training.learning_rate)
    )
    
    return model, optimizer
from omegaconf import DictConfig
from xai_gp.utils.training import train_model, prepare_data, initialize_model
from xai_gp.utils.evaluation import evaluate_model
from xai_gp.hyperparam_tuning.hyperparameter_optimization import (
    run_hyperparameter_optimization,
    get_best_model
)

import torch
import hydra
from xai_gp.utils.shift_analysis import run_shift_analysis


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, test_loader, input_shape = prepare_data(cfg, device)

    # Check if hyperparameter tuning is enabled
    if cfg.get("hyperparam_tuning", {}).get("enabled", False):
        print("Running hyperparameter optimization...")
        
        # Run optimization to find best parameters
        best_params = run_hyperparameter_optimization(
            cfg, train_loader, test_loader, input_shape, device
        )
        
        # Initialize model with best parameters
        model, optimizer = get_best_model(best_params, cfg, device)
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        train_model(model, train_loader, optimizer, cfg, best_params=best_params)
        evaluate_model(model, test_loader, cfg, best_params=best_params)
    else:
        # Standard training workflow
        model, optimizer = initialize_model(cfg, input_shape, device)
        train_model(model, train_loader, optimizer, cfg)
        evaluate_model(model, test_loader, cfg)
    
    run_shift_analysis(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()

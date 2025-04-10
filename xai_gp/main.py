from omegaconf import DictConfig, OmegaConf

from xai_gp.utils.statistical_comparison import statistical_comparison
from xai_gp.utils.training import train_model, prepare_data, initialize_model
from xai_gp.utils.evaluation import evaluate_model
from xai_gp.hyperparam_tuning.hyperparameter_optimization import (
    run_hyperparameter_optimization,
    get_best_model
)

import wandb
import torch
import hydra
from xai_gp.utils.shift_analysis import run_shift_analysis


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, test_loader, input_shape = prepare_data(cfg, device)
    
    is_tuning = cfg.get("hyperparam_tuning", {}).get("enabled", False)
    tuning_mode = "optimized" if is_tuning else 'standard'
    wandb_name = f"{cfg.model.type}_{cfg.data.name}_{tuning_mode}"

    compare_all = cfg.get("compare_all", {}).get("enabled", False)

    # Check if hyperparameter tuning is enabled
    if is_tuning:
        print("Running hyperparameter optimization...")
        
        # Run optimization to find best parameters
        best_params = run_hyperparameter_optimization(
            cfg, train_loader, val_loader, input_shape, device
        )
        
        # Initialize model with best parameters
        model, optimizer = get_best_model(best_params, cfg, device)
        
        wandb.init(
            project="uncertainty-estimation-gp",
            entity="im2latex-replicate",
            config=best_params,
            name=wandb_name,
        )
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        train_model(model, train_loader, optimizer, cfg, best_params=best_params, val_loader=val_loader)
        metrics = evaluate_model(model, test_loader, cfg, best_params=best_params)
    elif compare_all:
        wandb.init(
            project="uncertainty-estimation-gp",
            entity="im2latex-replicate",
            name=wandb_name,
        )

        statistical_comparison(cfg, train_loader, val_loader, test_loader, input_shape, device)

    else:
        wandb_config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        
        # Let's not log unoptimized hyperparameters to wandb
        wandb.init(
            project="uncertainty-estimation-gp",
            entity="im2latex-replicate",
            config=wandb_config,
            name=wandb_name,
            mode='offline',
        )

        # Standard training workflow
        model, optimizer = initialize_model(cfg, input_shape, device)
        train_model(model, train_loader, optimizer, cfg, val_loader=val_loader)
        metrics = evaluate_model(model, test_loader, cfg, plotting=True)
    
    wandb.log(metrics)
    
    if cfg.data.task_type == "regression":
        run_shift_analysis(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()

from omegaconf import DictConfig
from xai_gp.utils.training import train_model, prepare_data, initialize_model
from xai_gp.utils.evaluation import evaluate_model, is_gp_model

import torch
import hydra


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, test_loader, input_shape = prepare_data(cfg, device)

    # Initialize model
    model, optimizer = initialize_model(cfg, input_shape, device)

    # Train and evaluate model
    train_model(model, train_loader, optimizer, cfg)
    evaluate_model(model, test_loader, cfg)


if __name__ == "__main__":
    main()

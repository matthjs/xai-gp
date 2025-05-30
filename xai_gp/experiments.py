"""
This is a separate script for the ablation experiments. The main experiment of the paper are ran
through `main.py`.
"""

from matplotlib.ticker import FormatStrFormatter
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from xai_gp.models.gp import DeepGPModel, DSPPModel, fit_gp
from xai_gp.utils.training import prepare_data
from xai_gp.utils.evaluation import evaluate_model
import torch
import hydra


def ablation_inducing(cfg) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, input_shape = prepare_data(cfg, device)

    # Consider a single layer GP
    hidden_layer_config = [
        {
            # Warning: for simplicity we assume 2 classes but if needed this should be extended
            'output_dims': 2 if cfg.data.task_type == "classification" else None,
            'mean_type': 'constant'  # Constant mean for final layer
        }
    ]

    inducing_points = [32, 64, 128, 256, 512]
    constructors = [DeepGPModel, DSPPModel]
    model_names = ["DeepGP", "DSPP"]

    results = {name: {"x": [], "y": []} for name in model_names}

    for constructor, model_name in zip(constructors, model_names):
        for num_inducing in inducing_points:
            print(f"Training {model_name} with {num_inducing} inducing points...")
            model = constructor(
                input_dim=input_shape[-1],
                hidden_layers_config=hidden_layer_config,
                num_inducing_points=num_inducing,
                classification=cfg.data.task_type == "classification"
            ).to(device)

            optimizer = getattr(torch.optim, cfg.training.optimizer)(
                model.parameters(),
                lr=cfg.training.learning_rate
            )

            fit_gp(model, train_loader, cfg.training.num_epochs, optimizer, gp_mode=model_name)

            metrics = evaluate_model(model, test_loader, cfg)
            nll = metrics["nll"]

            results[model_name]["x"].append(num_inducing)
            results[model_name]["y"].append(nll)

    plt.figure(figsize=(8, 6))
    for name, data in results.items():
        plt.plot(data["x"], data["y"], marker='o', label=name)

    plt.title("Effect of Inducing Points on Negative Log Likelihood")
    plt.xlabel("Number of Inducing Points")
    plt.ylabel("Negative Log Likelihood")
    plt.xscale("log", base=2)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.savefig("inducing_points_vs_nll.png", dpi=300)
    plt.savefig("inducing_points_vs_nll.svg", dpi=300)


def ablation_layers(cfg) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, input_shape = prepare_data(cfg, device)

    num_inducing = 128
    depth_levels = [1, 2, 4, 8]

    constructors = [DeepGPModel, DSPPModel]
    model_names = ["DeepGP", "DSPP"]
    results = {name: {"x": [], "y": []} for name in model_names}

    for constructor, model_name in zip(constructors, model_names):
        for depth in depth_levels:
            print(f"Training {model_name} with {depth} layer(s)...")

            hidden_layer_config = [
                {"output_dims": 1, "mean_type": "linear"} for _ in range(depth - 1)
            ] + [{"output_dims": 2 if cfg.data.task_type == "classification" else None, "mean_type": "constant"}]

            model = constructor(
                input_dim=input_shape[-1],
                hidden_layers_config=hidden_layer_config,
                num_inducing_points=num_inducing,
                classification=cfg.data.task_type == "classification"
            ).to(device)

            optimizer = getattr(torch.optim, cfg.training.optimizer)(
                model.parameters(),
                lr=cfg.training.learning_rate
            )

            fit_gp(model, train_loader, cfg.training.num_epochs, optimizer, gp_mode=model_name)

            metrics = evaluate_model(model, test_loader, cfg)
            nll = metrics["nll"]

            results[model_name]["x"].append(depth)
            results[model_name]["y"].append(nll)

    plt.figure(figsize=(8, 6))
    for name, data in results.items():
        plt.plot(data["x"], data["y"], marker='o', label=name)

    plt.title("Effect of Depth on Negative Log Likelihood (1 unit per hidden layer)")
    plt.xlabel("Number of Layers")
    plt.ylabel("Negative Log Likelihood")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.savefig("depth_vs_nll.png", dpi=300)
    plt.savefig("depth_vs_nll.svg", dpi=300)


@hydra.main(version_base=None, config_path="../conf", config_name="abl_config")
def main(cfg: DictConfig) -> None:
    if cfg.experiment == 'ablation_inducing':
        ablation_inducing(cfg)
    elif cfg.experiment == 'ablation_layers':
        ablation_layers(cfg)
    else:
        raise ValueError(f"No valid experiment type")


if __name__ == "__main__":
    main()

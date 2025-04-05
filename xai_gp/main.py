import torch
import pandas as pd
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xai_gp.models.gp import DeepGPModel, DSPPModel, fit_gp

from xai_gp.models.ensemble import (
    DeepEnsembleRegressor,
    train_ensemble_regression,
    train_ensemble_classification,
    TwoHeadMLP
)

from omegaconf import DictConfig

from xai_gp.evaluation.calibration import (
    regressor_calibration_curve,
    regressor_calibration_error,
    classifier_calibration_curve
)


def prepare_data(cfg, device):
    """Load and preprocess the dataset."""
    # Load dataset
    dataset_path = cfg.data.path
    data = pd.read_csv(dataset_path)

    # Split into features and target
    X = data.iloc[:, cfg.data.feature_cols].values
    y = data.iloc[:, cfg.data.target_col].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=cfg.data.test_size, random_state=cfg.random_seed
    )

    # Convert to PyTorch tensors
    train_x = torch.FloatTensor(X_train).to(device)
    train_y = torch.FloatTensor(y_train).to(device)
    test_x = torch.FloatTensor(X_test).to(device)
    test_y = torch.FloatTensor(y_test).to(device)

    # Print dataset information
    print(f"Input features: {train_x.shape[1]}")
    print(f"Training samples: {train_x.shape[0]}")
    print(f"Test samples: {test_x.shape[0]}")

    # Create data loaders
    batch_size = cfg.training.batch_size
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.training.shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_x.shape


def is_gp_model(model):
    """Check if the model/class is a GP."""
    is_instance = isinstance(model, (DeepGPModel, DSPPModel))
    is_class = model in (DeepGPModel, DSPPModel)
    return is_instance or is_class


def initialize_model(cfg, input_shape, device):
    """Initialize the model based on configuration."""
    MODEL_TYPES = {
        "DeepGPModel": DeepGPModel,
        "DSPPModel": DSPPModel,
        "DeepEnsembleRegressor": DeepEnsembleRegressor
    }

    model_class = MODEL_TYPES.get(cfg.model.type)
    if (model_class is None):
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Check if the model is a GP model using is_gp_model function
    if is_gp_model(model_class):
        model = model_class(
            train_x_shape=input_shape,
            hidden_layers_config=cfg.model.hidden_layers,
            num_inducing_points=cfg.model.num_inducing_points,
        ).to(device)
    else:

        # For ensemble models, initialize with TwoHeadMLP as the base model
        input_dim = input_shape[1]
        output_dim = cfg.model.output_dim

        # Create a base model using TwoHeadMLP
        base_model = lambda: TwoHeadMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=cfg.model.hidden_layers
        )

        # Initialize the ensemble model with the base model
        model = model_class(
            model_fn=base_model,
            num_models=cfg.model.num_ensemble_models
        ).to(device)

    # Setup optimizer
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(),
        lr=cfg.training.learning_rate
    )

    return model, optimizer


def extract_predictions(model, batch_x):
    """Extract predictions based on model type."""
    if is_gp_model(model):
        # GP models return MultivariateNormal with mean and variance
        mvr = model(batch_x)
        return mvr.mean, mvr.variance
    else:
        # Ensemble models return mean and variance directly
        mean, variance = model(batch_x)
        return mean, variance


def plot_calibration_curve(conf, acc, title="Calibration Curve", relative_save_path='calibration_curve.png'):
    """
    Plot the calibration curve for regression uncertainty as a histogram.
    
    Args:
        conf: Confidence values (x-axis) - these are confidence levels (alphas)
        acc: Accuracy values (y-axis) - these are empirical probabilities of intervals
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(10, 8))

    # Plot the ideal calibration line (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")

    # Calculate the width of bars to cover the entire [0,1] range
    # Adjust bar positions to cover the entire range
    if len(conf) > 1:
        bin_width = 1.0 / (len(conf))
    else:
        bin_width = 0.05

    # Adjust bar positions so the first bar starts at 0 and the last ends at 1
    adjusted_conf = np.linspace(bin_width / 2, 1.0 - bin_width / 2, len(conf))

    # Plot bars to fill the entire [0,1] range
    plt.bar(adjusted_conf, acc, width=bin_width, alpha=0.7,
            edgecolor='black', linewidth=1, label="Model calibration")

    # Plots the confidence line
    plt.plot(adjusted_conf, acc, 'ro-', linewidth=2, markersize=6)

    # Add a horizontal line at y=0 to make the bars look connected to the axis
    plt.axhline(y=0, color='black', linewidth=0.5)

    plt.xlabel('Confidence Level')
    plt.ylabel('Empirical Coverage Probability')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Set limits to make the plot clearer
    plt.ylim([0, 1.00])
    plt.xlim([0, 1.00])

    if relative_save_path:
        save_path = os.path.join('../results', relative_save_path)
        # Make results directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Saving calibration curve plot to: {save_path}")
        plt.savefig(save_path)

    plt.show()


def train_and_evaluate(model, train_loader, test_loader, optimizer, cfg):
    """Train the model and evaluate its performance."""
    # Train model
    num_epochs = cfg.training.num_epochs
    print(f"Training for {num_epochs} epochs...")

    # Train based on model type
    if is_gp_model(model):
        beta = cfg.model.get('beta', None)  # Get beta if it exists, otherwise set to None
        fit_gp(model, train_loader, num_epochs, optimizer, gp_mode=cfg.model.gp_mode, beta=beta)
    else:
        # For ensemble models, use the appropriate training function based on task type
        if cfg.data.task_type == "regression":
            train_ensemble_regression(model, train_loader, num_epochs, cfg.training.learning_rate)
        elif cfg.data.task_type == "classification":
            train_ensemble_classification(model, train_loader, num_epochs, cfg.training.learning_rate)
        else:
            raise ValueError(f"Unknown task type: {cfg.data.task_type}. Must be 'regression' or 'classification'.")

    # Evaluate on test set
    model.eval()
    all_means = []
    all_variances = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Forward pass and extract predictions based on model type
            means, variances = extract_predictions(model, batch_x)

            all_means.append(means)
            all_variances.append(variances)
            all_targets.extend(batch_y.cpu().tolist())

        # Convert test targets to tensor
        test_targets = torch.tensor(all_targets).cpu()

        # Concatenate means and variances - handle dimensionality consistently
        if is_gp_model(model):
            test_means = torch.cat(all_means, dim=1).cpu()
            test_variances = torch.cat(all_variances, dim=1).cpu()
        else:
            test_means = torch.cat(all_means, dim=0).cpu()
            test_variances = torch.cat(all_variances, dim=0).cpu()

        # Calculate error metrics
        mse = torch.mean(abs(test_means - test_targets)).item()
        print(f"\nTest Results:")
        print(f"MAE: {mse:.4f}")

    # Convert to numpy due to library limitations
    test_means = test_means.numpy()
    test_variances = test_variances.numpy()

    # If regression, calculate calibration error
    if cfg.data.task_type == "regression":
        # Calculate standard deviations from variances
        test_stds = np.sqrt(test_variances)

        # Calculate calibration error
        calibration_error = regressor_calibration_error(test_means, test_stds, test_targets)
        print(f"Calibration error: {calibration_error:.4f}")

        # Get calibration curve data
        # conf represents confidence levels (alphas)
        # acc represents empirical coverage probabilities
        conf, acc = regressor_calibration_curve(test_means, test_stds, test_targets)

        # Plot the calibration curve
        plot_title = f"Calibration Curve for {cfg.model.type}"
        plot_calibration_curve(conf, acc, title=plot_title,
                               relative_save_path=f'calibration_{cfg.model.type}_{cfg.data.name}.png')
    else:
        pass  # Classification not implemented yet


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, test_loader, input_shape = prepare_data(cfg, device)

    # Initialize model
    model, optimizer = initialize_model(cfg, input_shape, device)
    train_and_evaluate(model, train_loader, test_loader, optimizer, cfg)


if __name__ == "__main__":
    main()

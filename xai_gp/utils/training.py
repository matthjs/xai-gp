from xai_gp.models.gp import fit_gp
from xai_gp.models.ensemble import train_ensemble_regression, train_ensemble_classification
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.models.ensemble import (
    DeepEnsembleRegressor,
    TwoHeadMLP,
    DeepEnsembleClassifier
)
from xai_gp.utils.evaluation import is_gp_model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.utils.data import random_split
import os


# Helper function to move data to the GPU
def collate_fn(batch, device='cuda'):
    data, target = zip(*batch)
    data = torch.stack(data).to(device)
    target = torch.tensor(target).to(device)
    return data, target


def prepare_data(cfg, device):
    dataset_path = cfg.data.path
    if cfg.data.name.lower() == "esr":
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Drop the 'Unnamed' column if it exists
        if 'Unnamed' in df.columns:
            df = df.drop(columns=['Unnamed'])

        # Separate features and labels
        X = df.drop(columns=['y']).values
        y = df['y'].values

        # Adjust labels: convert label 1 to 1 (seizure), and labels 2-5 to 0 (non-seizure)
        y = (y == 1).astype(int)

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)

        # Define split sizes
        total_size = len(dataset)
        train_size = int((1 - 2 * cfg.data.test_size) * total_size)
        val_size = test_size = int(cfg.data.test_size * total_size)

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        batch_size = cfg.training.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        input_shape = X_tensor.shape

        return train_loader, val_loader, test_loader, input_shape

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

    # Further split training data into training and validation sets
    val_size = cfg.data.test_size  # Use the same test_size for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=cfg.random_seed
    )

    # Convert to PyTorch tensors
    train_x = torch.FloatTensor(X_train).to(device)
    train_y = torch.FloatTensor(y_train).to(device)
    val_x = torch.FloatTensor(X_val).to(device)
    val_y = torch.FloatTensor(y_val).to(device)
    test_x = torch.FloatTensor(X_test).to(device)
    test_y = torch.FloatTensor(y_test).to(device)

    # Print dataset information
    print(f"Input features: {train_x.shape[1]}")
    print(f"Training samples: {train_x.shape[0]}")
    print(f"Validation samples: {val_x.shape[0]}")
    print(f"Test samples: {test_x.shape[0]}")

    # Create data loaders
    batch_size = cfg.training.batch_size
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.training.shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_x.shape


def initialize_model(cfg, input_shape, device):
    """Initialize the model based on configuration."""
    MODEL_TYPES = {
        "DeepGPModel": DeepGPModel,
        "DSPPModel": DSPPModel,
        "DeepEnsembleRegressor": DeepEnsembleRegressor,
        "DeepEnsembleClassifier": DeepEnsembleClassifier,
    }

    model_class = MODEL_TYPES.get(cfg.model.type)
    if (model_class is None):
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Check if the model is a GP model using is_gp_model function
    input_dim = input_shape[1]

    if is_gp_model(model_class):
        is_classification = cfg.data.task_type == "classification"
        model = model_class(
            input_dim=input_dim,
            hidden_layers_config=cfg.model.hidden_layers,
            num_inducing_points=cfg.model.num_inducing_points,
            classification=is_classification,
        ).to(device)
    else:
        # Create a base model using TwoHeadMLP
        base_model = lambda: TwoHeadMLP(
            input_dim=input_dim,
            output_dim=cfg.model.output_dim,
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


def train_model(model, train_loader, optimizer, cfg, best_params=None, val_loader=None):
    """Train the model."""
    # Use hyperparameter-optimized values if provided, otherwise use config values
    num_epochs = best_params.get("num_epochs", cfg.training.num_epochs) if best_params else cfg.training.num_epochs
    print(f"Training for {num_epochs} epochs...")

    if isinstance(model, (DeepGPModel, DSPPModel)):
        # For GP models, use beta from best_params if available
        beta = best_params.get("beta", cfg.model.get('beta', None)) if best_params else cfg.model.get('beta', None)
        gp_mode = best_params.get("gp_mode", cfg.model.gp_mode) if best_params else cfg.model.gp_mode
        loss = fit_gp(model, train_loader, num_epochs, optimizer, gp_mode=gp_mode, beta=beta, val_loader=val_loader)
    else:
        learning_rate = best_params.get("lr", cfg.training.learning_rate) if best_params else cfg.training.learning_rate
        if cfg.data.task_type == "regression":
            loss = train_ensemble_regression(model, train_loader, num_epochs, learning_rate, val_loader=val_loader)
        elif cfg.data.task_type == "classification":
            loss = train_ensemble_classification(model, train_loader, num_epochs, learning_rate, val_loader=val_loader)
        else:
            raise ValueError(f"Unknown task type: {cfg.data.task_type}. Must be 'regression' or 'classification'.")

    # Save the model
    os.makedirs("../results/weights", exist_ok=True)
    model_path = f"../results/weights/{cfg.model.type}_{cfg.data.task_type}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return loss

from xai_gp.models.gp import fit_gp
from xai_gp.models.ensemble import train_ensemble_regression, train_ensemble_classification
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.utils.logging import log_training_start, log_training_end
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.models.ensemble import (
    DeepEnsembleRegressor,
    TwoHeadMLP
)
from xai_gp.utils.evaluation import is_gp_model

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

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
    input_dim = input_shape[1]
    
    if is_gp_model(model_class):
        model = model_class(
            input_dim=input_dim,
            hidden_layers_config=cfg.model.hidden_layers,
            num_inducing_points=cfg.model.num_inducing_points,
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

def train_model(model, train_loader, optimizer, cfg):
    """Train the model."""
    num_epochs = cfg.training.num_epochs
    print(f"Training for {num_epochs} epochs...")

    start_time = log_training_start(cfg.model.type, num_epochs, len(train_loader.dataset))

    if isinstance(model, (DeepGPModel, DSPPModel)):
        beta = cfg.model.get('beta', None)
        fit_gp(model, train_loader, num_epochs, optimizer, gp_mode=cfg.model.gp_mode, beta=beta)
    else:
        if cfg.data.task_type == "regression":
            train_ensemble_regression(model, train_loader, num_epochs, cfg.training.learning_rate)
        elif cfg.data.task_type == "classification":
            train_ensemble_classification(model, train_loader, num_epochs, cfg.training.learning_rate)
        else:
            raise ValueError(f"Unknown task type: {cfg.data.task_type}. Must be 'regression' or 'classification'.")

    log_training_end(start_time, 0, len(train_loader))  # Replace `0` with actual loss if available

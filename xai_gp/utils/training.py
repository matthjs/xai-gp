from xai_gp.models.gp import fit_gp
from xai_gp.models.ensemble import train_ensemble_regression, train_ensemble_classification
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.utils.logging import log_training_start, log_training_end
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
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import random_split




def prepare_data(cfg, device):
    dataset_path = cfg.data.path
    if cfg.data.name.lower() == "cifar100":
        print("dataset_path", dataset_path)
        # Define image transforms including normalization.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761]),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        # Load CIFAR-100 using torchvision.
        full_train_dataset = CIFAR100(root=cfg.data.path, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=cfg.data.path, train=False, download=True, transform=transform)

        # Split the training dataset into training and validation sets
        val_size = int(len(full_train_dataset) * cfg.data.test_size)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Helper function to move data to the GPU
        def collate_fn(batch):
            data, target = zip(*batch)
            data = torch.stack(data).to(device)
            target = torch.tensor(target).to(device)
            return data, target

        # Create data loaders with the custom collate function
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=cfg.training.shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_fn)

        # For CIFAR-100, the input shape is (3, 32, 32).
        input_shape = (3, 32, 32)
        
        print("CIFAR-100 dataset loaded.")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
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
    if cfg.data.task_type == "classification":
        input_dim = int(torch.tensor(input_shape).prod().item())
    else:
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


def train_model(model, train_loader, optimizer, cfg, best_params=None):
    """Train the model."""
    # Use hyperparameter-optimized values if provided, otherwise use config values
    num_epochs = best_params.get("num_epochs", cfg.training.num_epochs) if best_params else cfg.training.num_epochs
    print(f"Training for {num_epochs} epochs...")

    if isinstance(model, (DeepGPModel, DSPPModel)):
        # For GP models, use beta from best_params if available
        beta = best_params.get("beta", cfg.model.get('beta', None)) if best_params else cfg.model.get('beta', None)
        gp_mode = best_params.get("gp_mode", cfg.model.gp_mode) if best_params else cfg.model.gp_mode
        loss = fit_gp(model, train_loader, num_epochs, optimizer, gp_mode=gp_mode, beta=beta)
    else:
        learning_rate = best_params.get("lr", cfg.training.learning_rate) if best_params else cfg.training.learning_rate
        if cfg.data.task_type == "regression":
            loss = train_ensemble_regression(model, train_loader, num_epochs, learning_rate)
        elif cfg.data.task_type == "classification":
            loss = train_ensemble_classification(model, train_loader, num_epochs, learning_rate)
        else:
            raise ValueError(f"Unknown task type: {cfg.data.task_type}. Must be 'regression' or 'classification'.")

    return loss

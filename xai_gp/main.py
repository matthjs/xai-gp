import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xai_gp.models.gp.fitgp import fit_gp
from xai_gp.models.gp import DeepGPModel, DSPPModel
import hydra
from omegaconf import DictConfig


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
        "DSPPModel": DSPPModel
    }
    
    model_class = MODEL_TYPES.get(cfg.model.type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    model = model_class(
        train_x_shape=input_shape,
        hidden_layers_config=cfg.model.hidden_layers,
        num_inducing_points=cfg.model.num_inducing_points,
    ).to(device)
    
    # Setup optimizer
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(),
        lr=cfg.training.learning_rate
    )
    
    return model, optimizer


def train_and_evaluate(model, train_loader, test_loader, optimizer, cfg):
    """Train the model and evaluate its performance."""
    # Train model
    num_epochs = cfg.training.num_epochs
    print(f"Training for {num_epochs} epochs...")
    beta = cfg.model.get('beta', None)  # Get beta if it exists, otherwise set to None
    fit_gp(model, train_loader, num_epochs, optimizer, gp_mode=cfg.model.gp_mode, beta=beta)
    
    # Evaluate on test set
    model.eval()
    all_mvrs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Forward pass
            mvr = model(batch_x)
            
            all_mvrs.append(mvr)
            all_targets.extend(batch_y.cpu().tolist())
        
        # Convert test targets to tensor
        test_targets = torch.tensor(all_targets).cpu()
        
        # Extract and concatenate means and variances
        test_means = torch.cat([mvr.mean for mvr in all_mvrs], dim=1).cpu()
        test_variances = torch.cat([mvr.variance for mvr in all_mvrs], dim=1).cpu()
        
        # Calculate error metrics
        mse = torch.mean(abs(test_means - test_targets)).item()
        uncertainty = torch.sqrt(test_variances).mean().item()
        print(f"\nTest Results:")
        print(f"MAE: {mse:.4f}")
        print(f"Average uncertainty: {uncertainty:.4f}")
    
    return test_means, test_variances, mse, uncertainty


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, test_loader, input_shape = prepare_data(cfg, device)
    
    # Initialize model
    model, optimizer = initialize_model(cfg, input_shape, device)
    
    # Train and evaluate
    predictions, variances, mse, uncertainty = train_and_evaluate(model, train_loader, test_loader, optimizer, cfg)

if __name__ == "__main__":
    main()
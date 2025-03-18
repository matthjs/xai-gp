import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xai_gp.models.gp.fitgp import fit_gp
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

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
    train_x = torch.FloatTensor(X_train)
    train_y = torch.FloatTensor(y_train)
    test_x = torch.FloatTensor(X_test)
    test_y = torch.FloatTensor(y_test)
    
    # Print dataset information
    print(f"Input features: {train_x.shape[1]}")
    print(f"Training samples: {train_x.shape[0]}")
    print(f"Test samples: {test_x.shape[0]}")

    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    batch_size = cfg.training.batch_size
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.training.shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model based on config
    model_class = globals()[cfg.model.type]
    model = model_class(
        train_x_shape=train_x.shape,
        hidden_layers_config=cfg.model.hidden_layers,
        num_inducing_points=cfg.model.num_inducing_points,
    ).to(device)

    # Setup optimizer
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(), 
        lr=cfg.training.learning_rate
    )
    
    # Train model
    num_epochs = cfg.training.num_epochs
    print(f"Training for {num_epochs} epochs...")
    fit_gp(model, train_loader, num_epochs, optimizer, gp_mode=cfg.model.gp_mode)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        # Check if model returns one or two outputs (mean only or mean and variance)
        mvr = model(test_x)
        mean, var = mvr.mean, mvr.variance
        
        # Calculate error metrics
        mse = torch.mean(abs(mean - test_y)).item()
        uncertainty = torch.sqrt(var).mean().item()
        
        print(f"\nTest Results:")
        print(f"MAE: {mse:.4f}")
        print(f"Uncertainty (variance): {uncertainty:.4f}")    

if __name__ == "__main__":
    main()
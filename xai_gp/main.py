import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.deepgp import DeepGPModel
from models.fitgp import fit_gp
from xai_gp.models.deepsigma import DSPPModel

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CASP protein dataset
    dataset_path = "data/CASP.csv"
    data = pd.read_csv(dataset_path)
    
    # Split into features and target
    X = data.iloc[:, 1:].values  # F1 through F9 features
    y = data.iloc[:, 0].values   # RMSD target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
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

    batch_size = 1024
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Architecture optimized for CASP protein dataset
    hidden_layers_config = [
        {
            'output_dims': 32,  # First hidden layer with 32 units
            'mean_type': 'linear'  # Linear mean function
        },
        {
            'output_dims': 16,  # Second hidden layer with 16 units
            'mean_type': 'linear'  # Linear mean function
        },
        {
            'output_dims': None,  # Output layer (None for univariate regression)
            'mean_type': 'constant'  # Constant mean for final layer
        }
    ]

    model = DSPPModel(
        train_x_shape=train_x.shape,  # Shape includes batch size and 9 features
        hidden_layers_config=hidden_layers_config,
        num_inducing_points=100,  # Inducing points per layer
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Train for more epochs for better convergence
    num_epochs = 10
    print(f"Training for {num_epochs} epochs...")
    fit_gp(model, train_loader, num_epochs, optimizer, gp_mode='DSPP')
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        # Check if model returns one or two outputs (mean only or mean and variance)
        mvr = model(test_x)
        mean, var = mvr.mean, mvr.variance
        
        # Calculate error metrics
        mse = torch.mean((mean - test_y) ** 2).item()
        rmse = torch.sqrt(torch.mean((mean - test_y) ** 2)).item()
        
        # Calculate R² score
        ss_total = torch.sum((test_y - test_y.mean()) ** 2)
        ss_residual = torch.sum((test_y - mean) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        print(f"\nTest Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2.item():.4f}")
        
        # Print uncertainty
        uncertainty = torch.sqrt(var).mean().item()
        print(f"Average prediction uncertainty: {uncertainty:.4f}")
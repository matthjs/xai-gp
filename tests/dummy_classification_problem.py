import torch
from torch.utils.data import TensorDataset, DataLoader
from xai_gp.models.deepsigma import DSPPModel
from gpytorch.likelihoods import BernoulliLikelihood  # For binary classification
import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
from xai_gp.models.fitgp import fit_gp

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate moons dataset (non-linearly separable)
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_test, y_test = make_moons(n_samples=200, noise=0.2, random_state=42)

    # Convert to PyTorch tensors
    train_x = torch.from_numpy(X).float()
    train_y = torch.from_numpy(y).long()  # Class indices (0 or 1)
    test_x = torch.from_numpy(X_test).float()
    test_y = torch.from_numpy(y_test).long()

    # Move data to GPU (if available)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    # Create datasets and data loaders
    batch_size = 64
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model configuration
    hidden_layers_config = [
        {'output_dims': 2, 'mean_type': 'linear'},  # First hidden layer
        {'output_dims': 2, 'mean_type': 'linear'},  # Second hidden layer
        {'output_dims': 2, 'mean_type': 'constant'}  # Output layer (2 classes)
    ]

    # Initialize DSPP model for classification
    model = DSPPModel(
        train_x_shape=train_x.shape,  # Shape of input data (1000, 3)
        hidden_layers_config=hidden_layers_config,
        num_inducing_points=128,  # Inducing points per layer
        classification=True
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    fit_gp(model, train_loader, 10, optimizer, gp_mode='DSPP')

    # Evaluate the model
    # This is more explicit but can also use model.posterior(test_x, apply_likelihood=True)
    # and then do probs.argmax(dim=-1).
    model.eval()
    with torch.no_grad():
        latent_dist = model(test_x)  # Latent function values
        pred_dist = model.likelihood(latent_dist)  # Class probabilities
        probs = pred_dist.probs.mean(dim=0)  # Average over samples
        pred_classes = probs.argmax(dim=-1)  # Predicted classes

    # Compute accuracy
    accuracy = (pred_classes == test_y).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
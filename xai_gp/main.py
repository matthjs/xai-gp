import torch
from torch.utils.data import TensorDataset, DataLoader
from models.deepgp import DeepGPModel
from models.fitgp import fit_gp
from xai_gp.models.deepsigma import DSPPModel

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_x = torch.randn(1000, 3)  # 1000 samples, 3 input dimensions
    train_y = torch.sin(train_x).sum(dim=1) + 0.1 * torch.randn(1000)
    test_x = torch.randn(200, 3)
    test_y = torch.sin(test_x).sum(dim=1) + 0.1 * torch.randn(200)

    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    batch_size = 64
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    hidden_layers_config = [
        {
            'output_dims': 5,  # First hidden layer with 5 units
            'mean_type': 'linear'  # Linear mean function for hidden layer
        },
        {
            'output_dims': None,  # Last layer (output) - None for univariate regression
            'mean_type': 'constant'  # Constant mean for final layer
        }
    ]

    # Or DeepGPModel
    model = DSPPModel(
        train_x_shape=train_x.shape,  # Shape of input data (1000, 3)
        hidden_layers_config=hidden_layers_config,
        num_inducing_points=128,  # Inducing points per layer
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    fit_gp(model, train_loader, 10, optimizer, gp_mode='DSPP')



import torch
from torch.utils.data import Dataset, DataLoader

from xai_gp.models.ensemble.deepensembleregressor import DeepEnsembleRegressor
from xai_gp.models.ensemble.fitensemble import train_ensemble_regression
from xai_gp.models.ensemble.twoheadmlp import TwoHeadMLP


# Regression Test Dataset
class SyntheticRegressionDataset(Dataset):
    def __init__(self, num_samples=100):
        self.x = torch.randn(num_samples, 2)
        self.y = 2 * self.x + 3 + 0.1 * torch.randn(num_samples, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Regression Tests
def test_deep_ensemble_regressor_fit():
    # Setup
    def model_fn_reg():
        return TwoHeadMLP(input_dim=2, output_dim=2)

    ensemble_reg = DeepEnsembleRegressor(model_fn=model_fn_reg, num_models=5)
    train_loader = DataLoader(SyntheticRegressionDataset(100), batch_size=10)

    # Train
    train_ensemble_regression(ensemble_reg, train_loader, num_epochs=15, lr=0.05)

    # Test
    test_data = SyntheticRegressionDataset(20)
    test_x, test_y = test_data.x, test_data.y

    with torch.no_grad():
        mean, var = ensemble_reg(test_x)
        mse = torch.mean((mean - test_y) ** 2).item()
        assert mse < 0.5, "MSE should be low after training"
        assert torch.all(var > 1e-20), "Variances must be positive"


def test_regression_uncertainty_disentangle():
    ensemble_reg = DeepEnsembleRegressor(
        model_fn=lambda: TwoHeadMLP(1, 1),
        num_models=5
    )
    test_x = torch.randn(5, 1)

    mean, var, ale_var, epi_var = ensemble_reg(test_x, disentangle_uncertainty=True)
    assert ale_var.shape == (5, 1), "Aleatoric variance shape mismatch"
    assert epi_var.shape == (5, 1), "Epistemic variance shape mismatch"
    assert torch.all(ale_var > 0), "Aleatoric variance must be positive"
    assert torch.all(epi_var >= 0), "Epistemic variance cannot be negative"
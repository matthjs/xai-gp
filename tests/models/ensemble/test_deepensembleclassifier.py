import torch
from torch.utils.data import Dataset, DataLoader

from xai_gp.models.ensemble.deepensembleclassifier import DeepEnsembleClassifier
from xai_gp.models.ensemble.fitensemble import train_ensemble_classification
from xai_gp.models.ensemble.twoheadmlp import TwoHeadMLP


# Classification Test Dataset
class SyntheticClassificationDataset(Dataset):
    def __init__(self, num_samples=100):
        self.x = torch.randn(num_samples, 2)
        self.y = (self.x[:, 0] + self.x[:, 1] > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_deep_ensemble_classifier_fit():
    # Setup
    def model_fn_clf():
        return TwoHeadMLP(input_dim=2, output_dim=2)

    ensemble_clf = DeepEnsembleClassifier(model_fn=model_fn_clf, num_models=5)
    train_data = SyntheticClassificationDataset(100)
    train_loader = DataLoader(train_data, batch_size=10)

    # Train
    train_ensemble_classification(ensemble_clf, train_loader, num_epochs=15, lr=0.05)

    # Test
    test_data = SyntheticClassificationDataset(20)
    test_x, test_y = test_data.x, test_data.y

    with torch.no_grad():
        prob = ensemble_clf(test_x)
        pred = prob.argmax(dim=1)
        accuracy = (pred == test_y).float().mean().item()
        assert accuracy > 0.8, "Accuracy should improve after training"
        assert torch.allclose(prob.sum(dim=1), torch.ones(20), atol=1e-4), "Probabilities must sum to 1"


def test_classification_uncertainty_disaggregation():
    ensemble_clf = DeepEnsembleClassifier(
        model_fn=lambda: TwoHeadMLP(2, 2, [{'output_dims': 10}]),
        num_models=3
    )
    test_x = torch.randn(5, 2)

    prob, prob_ale, prob_epi = ensemble_clf(test_x, disentangle_uncertainty=True)
    assert torch.allclose(prob.sum(dim=1), torch.ones(5), atol=1e-4)
    assert torch.allclose(prob_ale.sum(dim=1), torch.ones(5), atol=1e-4)
    assert torch.allclose(prob_epi.sum(dim=1), torch.ones(5), atol=1e-4)
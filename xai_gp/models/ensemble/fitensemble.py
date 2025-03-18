
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from xai_gp.models.ensemble.deepensembleclassifier import DeepEnsembleClassifier, sampling_softmax
from xai_gp.models.ensemble.deepensembleregressor import DeepEnsembleRegressor


# Training function for regression
def train_ensemble_regression(ensemble: DeepEnsembleRegressor,
                              train_loader,
                              num_epochs: int,
                              lr: float = 0.01) -> None:    # Make hyperparameters more flexible here
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss = nn.GaussianNLLLoss()

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            for model, optimizer in zip(ensemble.models, optimizers):
                optimizer.zero_grad()
                mean, var = model(x_batch)
                loss = loss(mean, y_batch, var)
                loss.backward()
                optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")


# Training function for classification
def train_ensemble_classification(ensemble: DeepEnsembleClassifier,
                                  train_loader,
                                  num_epochs: int,
                                  lr: float = 0.01) -> None:
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            for model, optimizer in zip(ensemble.models, optimizers):
                optimizer.zero_grad()
                mean, var = model(x_batch)
                prob = sampling_softmax(mean, var)
                loss = loss(prob, y_batch)
                loss.backward()
                optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
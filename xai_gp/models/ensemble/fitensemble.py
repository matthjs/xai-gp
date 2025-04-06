import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.distributions import MultivariateNormal
from xai_gp.models.ensemble.deepensembleclassifier import DeepEnsembleClassifier, sampling_softmax
from xai_gp.models.ensemble.deepensembleregressor import DeepEnsembleRegressor
from xai_gp.utils.logging import log_training_start, log_epoch_stats, log_training_end


# Training function for regression
def train_ensemble_regression(ensemble: DeepEnsembleRegressor,
                              train_loader,
                              num_epochs: int,
                              lr: float = 0.01) -> float:    # Make hyperparameters more flexible here
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss_fn = nn.GaussianNLLLoss()
    
    num_samples = len(train_loader.dataset)
    start_time = log_training_start("Ensemble Regression", num_epochs, num_samples)

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for x_batch, y_batch in train_loader:
            batch_loss = 0.0
            for model, optimizer in zip(ensemble.models, optimizers):
                optimizer.zero_grad()
                mean, var = model(x_batch)
                loss = loss_fn(mean, y_batch, var)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            
            # Average loss across all ensemble members
            batch_loss /= len(ensemble.models)
            epoch_loss += batch_loss
        
        # Log epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss
        log_epoch_stats(epoch, num_epochs, epoch_loss, len(train_loader), epoch_start_time)
    
    # Log final training summary
    log_training_end(start_time, epoch_loss, len(train_loader))
    return total_loss / num_epochs


# Training function for classification
def train_ensemble_classification(ensemble: DeepEnsembleClassifier,
                                  train_loader,
                                  num_epochs: int,
                                  lr: float = 0.01) -> float:
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss_fn = nn.CrossEntropyLoss()
    
    num_samples = len(train_loader.dataset)
    start_time = log_training_start("Ensemble Classification", num_epochs, num_samples)

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for x_batch, y_batch in train_loader:
            batch_loss = 0.0
            for model, optimizer in zip(ensemble.models, optimizers):
                optimizer.zero_grad()
                mean, var = model(x_batch)
                logits = MultivariateNormal(mean, torch.diag_embed(var)).rsample()
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            
            # Average loss across all ensemble members
            batch_loss /= len(ensemble.models)
            epoch_loss += batch_loss
        
        # Log epoch statistics
        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss
        log_epoch_stats(epoch, num_epochs, epoch_loss, len(train_loader), epoch_start_time)
    
    # Log final training summary
    log_training_end(start_time, epoch_loss, len(train_loader))
    return total_loss / num_epochs
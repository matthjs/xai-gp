import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.distributions import MultivariateNormal
from xai_gp.models.ensemble.deepensembleclassifier import DeepEnsembleClassifier, sampling_softmax
from xai_gp.models.ensemble.deepensembleregressor import DeepEnsembleRegressor
from xai_gp.utils.logging import log_training_start, log_epoch_stats, log_training_end


def run_ensemble_regression_epoch(ensemble, data_loader, loss_fn, optimizers=None, is_training=True):
    """
    Run one epoch of training or evaluation on a regression ensemble.
    When is_training=True, performs gradient updates with optimizer.
    When is_training=False, just computes loss without gradients.
    """
    if is_training:
        ensemble.train()
    else:
        ensemble.eval()
    
    total_loss = 0.0
    
    with torch.no_grad() if not is_training else torch.enable_grad():
        for x_batch, y_batch in data_loader:
            batch_loss = 0.0
            
            if is_training and optimizers:
                # Training mode with optimization
                for model, optimizer in zip(ensemble.models, optimizers):
                    optimizer.zero_grad()
                    mean, var = model(x_batch)
                    loss = loss_fn(mean, y_batch, var)
                    
                    if is_training:
                        loss.backward()
                        optimizer.step()
                        
                    batch_loss += loss.item()
                batch_loss /= len(ensemble.models)
            else:
                # Evaluation mode or no optimizers provided
                mean, var = ensemble(x_batch)
                batch_loss = loss_fn(mean, y_batch, var).item()
            
            total_loss += batch_loss
    
    return total_loss / len(data_loader)


def run_ensemble_classification_epoch(ensemble, data_loader, loss_fn, jitter=1e-3, optimizers=None, is_training=True):
    """
    Run one epoch of training or evaluation on a classification ensemble.
    When is_training=True, performs gradient updates with optimizer.
    When is_training=False, just computes loss without gradients.
    """
    if is_training:
        ensemble.train()
    else:
        ensemble.eval()
    
    total_loss = 0.0
    
    with torch.no_grad() if not is_training else torch.enable_grad():
        for x_batch, y_batch in data_loader:
            batch_loss = 0.0
            
            # Process each model in the ensemble
            for i, model in enumerate(ensemble.models):
                # Compute predictions and loss
                mean, var = model(x_batch)
                logits = MultivariateNormal(mean, torch.diag_embed(var + jitter)).rsample()
                loss = loss_fn(logits, y_batch)
                
                # Apply optimization if in training mode
                if is_training and optimizers:
                    optimizer = optimizers[i]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Accumulate loss (as float value regardless of mode)
                batch_loss += loss.item() if is_training else loss.item()
            
            # Average loss across the ensemble
            batch_loss /= len(ensemble.models)
            total_loss += batch_loss
    
    return total_loss / len(data_loader)


# Training function for regression
def train_ensemble_regression(ensemble: DeepEnsembleRegressor,
                              train_loader,
                              num_epochs: int,
                              lr: float = 0.01,
                              val_loader=None) -> float:
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss_fn = nn.GaussianNLLLoss()
    
    num_samples = len(train_loader.dataset)
    start_time = log_training_start("Ensemble Regression", num_epochs, num_samples)

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Run training epoch
        avg_epoch_loss = run_ensemble_regression_epoch(ensemble, train_loader, loss_fn, optimizers, is_training=True)
        total_loss += avg_epoch_loss
        log_epoch_stats(epoch, num_epochs, avg_epoch_loss, len(train_loader), epoch_start_time)
        
        # Run validation epoch if validation data is provided
        if val_loader is not None:
            avg_val_loss = run_ensemble_regression_epoch(ensemble, val_loader, loss_fn, is_training=False)
            log_epoch_stats(epoch, num_epochs, avg_val_loss, len(val_loader), epoch_start_time, type='val')
    
    # Log final training summary
    log_training_end(start_time, avg_epoch_loss, len(train_loader))
    return total_loss / num_epochs


# Training function for classification
def train_ensemble_classification(ensemble: DeepEnsembleClassifier,
                                  train_loader,
                                  num_epochs: int,
                                  lr: float = 0.01,
                                  val_loader=None) -> float:
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss_fn = nn.CrossEntropyLoss()
    jitter = 1e-2  # Small constant to avoid numerical instability
    
    num_samples = len(train_loader.dataset)
    start_time = log_training_start("Ensemble Classification", num_epochs, num_samples)

    total_loss = 0.0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Run training epoch
        avg_epoch_loss = run_ensemble_classification_epoch(ensemble, train_loader, loss_fn, jitter, optimizers, is_training=True)
        total_loss += avg_epoch_loss
        log_epoch_stats(epoch, num_epochs, avg_epoch_loss, len(train_loader), epoch_start_time)
        
        # Run validation epoch if validation data is provided
        if val_loader is not None:
            avg_val_loss = run_ensemble_classification_epoch(ensemble, val_loader, loss_fn, jitter, is_training=False)
            log_epoch_stats(epoch, num_epochs, avg_val_loss, len(val_loader), epoch_start_time, type='val')
    
    # Log final training summary
    log_training_end(start_time, avg_epoch_loss, len(train_loader))
    return total_loss / num_epochs
import torch
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL, ExactMarginalLogLikelihood
import time
from xai_gp.models.gp.gpbase import GPytorchModel
from gpytorch.mlls import DeepPredictiveLogLikelihood
from xai_gp.utils.logging import log_training_start, log_epoch_stats, log_training_end


def run_gp_epoch(model, data_loader, mll, optimizer=None, is_training=True):
    """
    Run one epoch of training or evaluation on a GP model.
    When is_training=True, performs gradient updates with optimizer.
    When is_training=False, just computes loss without gradients.
    """
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    # Determine the target device from the model
    device = next(model.parameters()).device
    
    with torch.no_grad() if not is_training else torch.enable_grad():
        for x_batch, y_batch in data_loader:
            # Move batches to the target device if they're not already there
            if x_batch.device != device:
                x_batch = x_batch.to(device)
            if y_batch.device != device:
                y_batch = y_batch.to(device)
                
            if is_training and optimizer:
                optimizer.zero_grad()
                
            output = model(x_batch)
            loss = -mll(output, y_batch).mean()
            
            if is_training and optimizer:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def fit_gp(model: GPytorchModel,
           data_loader,
           num_epochs: int,
           optimizer: torch.optim.Optimizer,
           gp_mode: str = 'DGP',
           beta: float = 0.05,
           val_loader=None,
           ) -> float:
    """
    Helper function for fitting gps to data.
    Make sure gp_mode corresponds to the GP model passed!
    """
    num_samples = len(data_loader.dataset)

    # Construct (marginal) log likelihood.
    if gp_mode == 'DeepGP':
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_samples))
    elif gp_mode == 'DSPP':
        mll = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=num_samples, beta=beta)
    elif gp_mode == 'SVGP':
        mll = VariationalELBO(model.likelihood, model.model, num_samples)
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

    start_time = log_training_start(gp_mode, num_epochs, num_samples)

    total_loss = 0.0
    # This optimizes the hyperparameters so noise variance, inducing points etc.
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Run training epoch
        avg_epoch_loss = run_gp_epoch(model, data_loader, mll, optimizer, is_training=True)
        total_loss += avg_epoch_loss
        log_epoch_stats(epoch, num_epochs, avg_epoch_loss, len(data_loader), epoch_start_time)
        
        # Run validation epoch if validation data is provided
        if val_loader is not None:
            avg_val_loss = run_gp_epoch(model, val_loader, mll, optimizer=None, is_training=False)
            log_epoch_stats(epoch, num_epochs, avg_val_loss, len(val_loader), epoch_start_time, type='val')

    # Log final training summary
    log_training_end(start_time, avg_epoch_loss, len(data_loader))

    return total_loss / num_epochs

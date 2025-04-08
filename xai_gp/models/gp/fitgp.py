import torch
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL, ExactMarginalLogLikelihood
import time
from xai_gp.models.gp.gpbase import GPytorchModel
from gpytorch.mlls import DeepPredictiveLogLikelihood
from xai_gp.utils.logging import log_training_start, log_epoch_stats, log_training_end


def fit_gp(model: GPytorchModel,
           data_loader,
           num_epochs: int,
           optimizer: torch.optim.Optimizer,
           gp_mode: str = 'DGP',
           beta: float = 0.05
           ) -> float:
    """
    Helper function for fitting gps to data.
    Make sure gp_mode corresponds to the GP model passed!
    """
    model.train()

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
    epoch_loss = 0
    # This optimizes the hyperparameters so noise variance, inducing points etc.
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            
            loss = -mll(output, y_batch)

            # Take the mean of the loss tensor 
            # Needed for DeepGP since a batch of losses is returned
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(data_loader)
        total_loss += avg_epoch_loss
        # Log epoch statistics
        log_epoch_stats(epoch, num_epochs, epoch_loss, len(data_loader), epoch_start_time)

    # Log final training summary
    log_training_end(start_time, epoch_loss, len(data_loader))

    return total_loss / num_epochs

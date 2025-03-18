import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from loguru import logger
import time
from xai_gp.models.gp.gpbase import GPytorchModel
from gpytorch.mlls import DeepPredictiveLogLikelihood


def fit_gp(model: GPytorchModel,
           data_loader,
           num_epochs: int,
           optimizer: torch.optim.Optimizer,
           gp_mode: str = 'DGP'
           ):
    """
    Helper function for fitting gps to data.
    Make sure gp_mode corresponds to the GP model passed!
    """
    model.train()

    num_samples = len(data_loader.dataset)

    # Construct (marginal) log likelihood.
    if gp_mode == 'DGP':
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_samples))
    elif gp_mode == 'DSPP':
        # TODO: beta is a hyperparameter of DSPP
        mll = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=num_samples, beta=0.05)
    elif gp_mode == 'SVGP':
        mll = VariationalELBO(model.likelihood, model.model, num_samples)
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

    logger.info(f"Training {gp_mode} model for {num_epochs} epochs on {num_samples} samples...")
    start_time = time.time()

    epoch_loss = 0
    # This optimizes the hyperparameters so noise variance, inducing points etc.
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(x_batch)

            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / len(data_loader)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_epoch_loss:.4f} - "
            f"Time: {epoch_time:.2f}s"
        )

    # Log final training summary
    total_time = time.time() - start_time
    logger.info(
        f"Training completed in {total_time:.2f}s. "
        f"Final Loss: {epoch_loss / len(data_loader):.4f}"
    )
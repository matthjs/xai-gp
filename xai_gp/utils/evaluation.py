import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import f1_score

from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.utils.calibration import (
    regressor_calibration_curve,
    regressor_calibration_error,
    classifier_calibration_curve,
    classifier_calibration_error,
)

def is_gp_model(model):
    """Check if the model/class is a GP."""
    is_instance = isinstance(model, (DeepGPModel, DSPPModel))
    is_class = model in (DeepGPModel, DSPPModel)
    is_string = model in ("DeepGPModel", "DSPPModel")
    return is_instance or is_class or is_string


def extract_predictions(model, batch_x, is_classification=False):
    """Extract predictions based on model type."""
    
    if is_classification:
        if is_gp_model(model):
            latent_dist = model(batch_x)  # Latent function values
            pred_dist = model.likelihood(latent_dist)  # Class probabilities
            probs = pred_dist.probs.mean(dim=0)
            return probs, None
        else:
            probs = model(batch_x)
            return probs, None
    else:
        if is_gp_model(model):
            mvr = model.posterior(batch_x, apply_likelihood=True)
            means, vars = mvr.mean, mvr.variance
            mean = means.mean(dim=0)
            var = vars.mean(dim=0)
            return mean, var
        else:
            mean, variance = model(batch_x)
            return mean, variance
    

def plot_calibration_curve(conf, acc, title="Calibration Curve", relative_save_path='calibration_curve.png'):
    """
    Plot the calibration curve for regression uncertainty as a histogram.
    
    Args:
        conf: Confidence values (x-axis) - these are confidence levels (alphas)
        acc: Accuracy values (y-axis) - these are empirical probabilities of intervals
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(10, 8))

    # Plot the ideal calibration line (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")

    # Calculate the width of bars to cover the entire [0,1] range
    # Adjust bar positions to cover the entire range
    if len(conf) > 1:
        bin_width = 1.0 / (len(conf))
    else:
        bin_width = 0.05

    # Adjust bar positions so the first bar starts at 0 and the last ends at 1
    adjusted_conf = np.linspace(bin_width / 2, 1.0 - bin_width / 2, len(conf))

    # Plot bars to fill the entire [0,1] range
    plt.bar(adjusted_conf, acc, width=bin_width, alpha=0.7,
            edgecolor='black', linewidth=1, label="Model calibration")

    # Plots the confidence line
    plt.plot(adjusted_conf, acc, 'ro-', linewidth=2, markersize=6)

    # Add a horizontal line at y=0 to make the bars look connected to the axis
    plt.axhline(y=0, color='black', linewidth=0.5)

    plt.xlabel('Confidence Level')
    plt.ylabel('Empirical Coverage Probability')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Set limits to make the plot clearer
    plt.ylim([0, 1.00])
    plt.xlim([0, 1.00])

    if relative_save_path:
        save_path = os.path.join('../results', relative_save_path)
        # Make results directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Saving calibration curve plot to: {save_path}")
        plt.savefig(save_path)
        
        wandb.log({"calibration_curve": wandb.Image(save_path)})


def evaluate_model(model, test_loader, cfg, best_params=None, plotting=False):
    """Evaluate the model's performance."""
    
    if cfg.data.task_type == "classification":
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                prob, _ = extract_predictions(model, batch_x, is_classification=True)  # This should now be a tensor of shape (batch_size, num_classes)
                all_probs.append(prob)
                all_targets.append(batch_y)
        
        test_probs = torch.cat(all_probs, dim=0)
        test_targets = torch.cat(all_targets, dim=0)
        
        # Calculate predicted class labels
        predicted = torch.argmax(test_probs, dim=1)
        accuracy = (predicted == test_targets).float().mean().item()
        
        # Compute Negative Log-Likelihood
        log_probs = torch.log(test_probs + 1e-10)
        nll = F.nll_loss(log_probs, test_targets)
        
        # Compute Brier Score: one-hot encode targets and compute squared difference.
        num_classes = test_probs.size(1)
        one_hot_targets = F.one_hot(test_targets, num_classes=num_classes).float()
        brier_score = torch.mean((test_probs - one_hot_targets) ** 2).item()
        
        # Send to CPU for calibration curve
        test_probs = test_probs.cpu().numpy()
        test_targets = test_targets.cpu().numpy()
        predicted = predicted.cpu().numpy()
        
        conf, acc = classifier_calibration_curve(predicted, test_targets, test_probs)
        error = classifier_calibration_error(predicted, test_targets, test_probs)
        
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"NLL Loss: {nll.item():.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'nll': nll.item(),
            'brier_score': brier_score,
            'calibration_error': error,
        }
        
        if best_params or plotting:
            # For the plot title, include information about whether we're using optimized parameters
            model_name = cfg.model.type
            plot_title = f"Calibration Curve for {model_name}"
            save_path = f'calibration_{model_name}_{cfg.data.name}.png'
            
            plot_calibration_curve(conf, acc, title=plot_title, relative_save_path=save_path)
        
        return metrics
    
    else:
        all_means = []
        all_variances = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                means, variances = extract_predictions(model, batch_x)
                all_means.append(means)
                all_variances.append(variances)
                all_targets.extend(batch_y.cpu().tolist())

            test_targets = torch.tensor(all_targets).cpu()

            test_means = torch.cat(all_means, dim=0).cpu()
            test_variances = torch.cat(all_variances, dim=0).cpu()

        test_means = test_means.numpy()
        test_variances = test_variances.numpy()

        test_stds = np.sqrt(test_variances)
        test_targets = test_targets.numpy()
        
        mae = np.mean(np.abs(test_means - test_targets))
        mse = np.mean((test_means - test_targets) ** 2)
        rmse = np.sqrt(mse)

        # Negative Log-Likelihood for Gaussian likelihood.
        # For simplicity I am adding it here but can be defined as a function
        nll = np.mean(
            0.5 * np.log(2 * np.pi * test_variances) + ((test_targets - test_means) ** 2) / (2 * test_variances)
        )

        conf, acc = regressor_calibration_curve(test_means, test_targets, test_stds)
        calibration_error = regressor_calibration_error(
            error_metric="mae",
            precomputed_conf=conf,
            precomputed_acc=acc
        )
        
        print(f"Calibration error: {calibration_error:.4f}")

        # Only plot for optimized values
        if best_params or plotting:
            # For the plot title, include information about whether we're using optimized parameters
            model_name = cfg.model.type
            plot_title = f"Calibration Curve for {model_name}"
            save_path = f'calibration_{model_name}_{cfg.data.name}.png'
            
            plot_calibration_curve(conf, acc, title=plot_title, relative_save_path=save_path)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'nll': nll,
            'calibration_error': calibration_error,
            'sharpness': np.mean(test_stds ** 2),  # Average variance
        }

        return metrics

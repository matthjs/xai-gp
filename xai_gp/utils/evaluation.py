import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.utils.calibration import (
    regressor_calibration_curve,
    regressor_calibration_error
)


def is_gp_model(model):
    """Check if the model/class is a GP."""
    is_instance = isinstance(model, (DeepGPModel, DSPPModel))
    is_class = model in (DeepGPModel, DSPPModel)
    return is_instance or is_class


def extract_predictions(model, batch_x):
    """Extract predictions based on model type."""
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


def evaluate_model(model, test_loader, cfg):
    """Evaluate the model's performance."""
    model.eval()
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

    if cfg.data.task_type == "regression":
        test_stds = np.sqrt(test_variances)
        test_targets = test_targets.numpy()
        
        mae = np.mean(np.abs(test_means - test_targets))
        print(f"Mean Absolute Error: {mae:.4f}")

        conf, acc = regressor_calibration_curve(test_means, test_targets, test_stds)
        calibration_error = regressor_calibration_error(
            error_metric="mae",
            precomputed_conf=conf,
            precomputed_acc=acc
        )
        
        print(f"Calibration error: {calibration_error:.4f}")

        plot_title = f"Calibration Curve for {cfg.model.type}"
        plot_calibration_curve(conf, acc, title=plot_title,
                               relative_save_path=f'calibration_{cfg.model.type}_{cfg.data.name}.png')
        
        metrics = {
            'mae': mae,
            'calibration_error': calibration_error,
            'sharpness': np.mean(test_stds ** 2),  # Average variance
        }
        
    return metrics

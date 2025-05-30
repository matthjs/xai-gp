import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from xai_gp.utils.evaluation import evaluate_model
from xai_gp.utils.training import collate_fn
from xai_gp.utils.shift import apply_shift
from xai_gp.utils.training import initialize_model, train_model

"""
To evaluate model calibration under distribution shift, we apply controlled 
input perturbations inspired by the framework of Ovadia et al. (2019).
"""


def plot_aggregated_boxplot(cfg, results, severity_levels, metric, ylabel):
    aggregated = {sev: [] for sev in severity_levels}
    # Iterate over each corruption type
    for corruption, sev_dict in results.items():
        # Iterate over each severity level and its corresponding result list
        for sev, entries in sev_dict.items():
            for entry in entries:
                # entry is now the dictionary containing the metrics
                aggregated[sev].append(entry[metric])
    data = [aggregated[sev] for sev in severity_levels]
    positions = list(range(len(severity_levels)))
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                    showfliers=True, whis=[0, 100])
    for box in bp['boxes']:
        box.set(facecolor='lightblue')
    ax.set_xticks(positions)
    ax.set_xticklabels(severity_levels)
    ax.set_xlim(-0.5, len(severity_levels) - 0.5)
    ax.set_xlabel("Shift Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Shift Severity (Aggregated Across Shift Types)")
    save_path = os.path.join('../results', 'shift_analysis', f'boxplot_{cfg.data.name}_{cfg.model.type}_{metric}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving aggregated box plot to: {save_path}")
    plt.savefig(save_path)
    # wandb.log({f"boxplot_{cfg.data.name}_{cfg.model.type}_{metric}": wandb.Image(save_path)})


def evaluate_under_shift(model, test_loader, cfg, batch_size, device, shift_type, severity):
    """
    Evaluate the model on shifted test data.
    
    For regression tasks, return MAE, calibration error and derived empirical coverage.
    """
    all_features = []
    all_labels = []
    for batch_x, batch_y in test_loader:
        all_features.append(batch_x)
        all_labels.append(batch_y)
    test_features = torch.cat(all_features, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    X_np = test_features.cpu().numpy()
    shifted_X_np = apply_shift(X_np, shift_type, severity)
    shifted_X_np = shifted_X_np.reshape(test_features.shape)
    shifted_test_tensor = torch.FloatTensor(shifted_X_np).to(device)
    shifted_test_dataset = TensorDataset(shifted_test_tensor, test_labels.cpu())
    shifted_test_loader = DataLoader(shifted_test_dataset, batch_size=batch_size,
                                     collate_fn=lambda batch: collate_fn(batch, device=device))
    metrics = evaluate_model(model, shifted_test_loader, cfg)

    if cfg.data.task_type == "classification":
        accuracy = metrics['accuracy']
        cal_error = metrics['calibration_error']
        return {"severity": severity, "accuracy": accuracy, "cal_error": cal_error}
    else:
        mae = metrics['mae']
        cal_error = metrics['calibration_error']
        return {"severity": severity, "mae": mae, "cal_error": cal_error}


def run_shift_analysis(train_loader, val_loader, test_loader, input_shape, cfg, device):
    """
    Run shift analysis for both regression and classification.
    Save the results to then combine the outputs from different models into one big grouped plot.
    """
    num_runs = cfg.shift_analysis.n_runs
    results = {}
    corruption_types = ["gaussian", "mask", "scaling", "permute", "outlier"]
    severity_levels = [0, 0.1, 0.2, 0.4, 0.6, 0.8]

    # Iterate over independent runs
    for run in range(num_runs):
        # Random seed
        seed = run
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

        # Initializing and training the model once per run
        model, optimizer = initialize_model(cfg, input_shape, device)
        train_model(model, train_loader, optimizer, cfg, val_loader=val_loader)

        # For each corruption type and severity, evaluate the already trained model
        for corruption in corruption_types:
            if corruption not in results:
                results[corruption] = {}
            for sev in severity_levels:
                if sev not in results[corruption]:
                    results[corruption][sev] = []
                metric_dict = evaluate_under_shift(model, test_loader, cfg, cfg.training.batch_size, device, corruption,
                                                   sev)
                results[corruption][sev].append(metric_dict)

                if cfg.data.task_type == "classification":
                    print(
                        f"Run: {run}, Corruption: {corruption}, Severity: {sev}, Accuracy: {metric_dict['accuracy']:.4f}, Cal_error: {metric_dict['cal_error']:.4f}")
                else:
                    print(
                        f"Run: {run}, Corruption: {corruption}, Severity: {sev}, MAE: {metric_dict['mae']:.4f}, Cal Error: {metric_dict['cal_error']:.4f}")

    # Save these results along with the model method for later combined analysis.
    save_path = os.path.join('../results', 'shift_analysis', f"{cfg.data.name}_{cfg.model.type}.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, results=results, method=cfg.model.type)
    print(f"Saved regression shift results for method {cfg.model.type} to {save_path}")

    if cfg.data.task_type == "classification":
        plot_aggregated_boxplot(cfg, results, severity_levels, metric="cal_error", ylabel="Empirical Calibration Error")
        plot_aggregated_boxplot(cfg, results, severity_levels, metric="accuracy", ylabel="Accuracy")
    else:
        plot_aggregated_boxplot(cfg, results, severity_levels, metric="cal_error", ylabel="Empirical Calibration Error")
        plot_aggregated_boxplot(cfg, results, severity_levels, metric="mae", ylabel="MAE")

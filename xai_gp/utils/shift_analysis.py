import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from xai_gp.utils.evaluation import evaluate_model

def plot_aggregated_boxplot(results, severity_levels, metric="cal_error", ylabel="Calibration Error"):
    aggregated = {sev: [] for sev in severity_levels}
    for shift in results:
        for entry in results[shift]:
            sev = entry["severity"]
            if sev in aggregated:
                aggregated[sev].append(entry[metric])
    
    data = [aggregated[sev] for sev in severity_levels]
    positions = list(range(len(severity_levels)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True, showfliers=True, whis=[0, 100])
    
    for box in bp['boxes']:
        box.set(facecolor='lightblue')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(severity_levels)
    ax.set_xlim(-0.5, len(severity_levels)-0.5)
    ax.set_xlabel("Shift Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Shift Severity (Aggregated Across Shift Types)")
    
    save_path = os.path.join('../results', 'shift_analysis', f'aggregated_boxplot_{metric}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving aggregated box plot to: {save_path}")
    plt.savefig(save_path)
    plt.show()

def plot_aggregated_boxplot_accuracy(results, severity_levels, metric="coverage", ylabel="Empirical Coverage"):
    aggregated = {sev: [] for sev in severity_levels}
    for shift in results:
        for entry in results[shift]:
            sev = entry["severity"]
            if sev in aggregated:
                aggregated[sev].append(entry[metric])
    
    data = [aggregated[sev] for sev in severity_levels]
    positions = list(range(len(severity_levels)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True, showfliers=True, whis=[0, 100])
    
    for box in bp['boxes']:
        box.set(facecolor='lightblue')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(severity_levels)
    ax.set_xlim(-0.5, len(severity_levels)-0.5)
    ax.set_xlabel("Shift Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Shift Severity (Aggregated Across Shift Types)")
    
    save_path = os.path.join('../results', 'shift_analysis', 'aggregated_boxplot_coverage.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving aggregated coverage box plot to: {save_path}")
    plt.savefig(save_path)
    plt.show()

def apply_shift(X, shift_type, severity):
    if shift_type == "gaussian":
        noise = np.random.normal(loc=0, scale=severity * np.std(X, axis=0), size=X.shape)
        return X + noise
    elif shift_type == "mask":
        mask = np.random.rand(*X.shape) < severity
        X_shifted = X.copy()
        X_shifted[mask] = 0
        return X_shifted
    elif shift_type == "scaling":
        scaling_factor = 1 + severity
        return X * scaling_factor
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")
    

def evaluate_under_shift(model, test_loader, cfg, batch_size, device, shift_type, severity, best_params=None):
    """
    Evaluate the model on shifted test data.
    
    Args:
        model: The trained model.
        test_loader (DataLoader): DataLoader containing test data.
        batch_size (int): Batch size for evaluation.
        device: Torch device.
        shift_type (str): The type of shift to apply ("gaussian", "mask", "scaling").
        severity (float): Severity level of the corruption.
    
    Returns:
        mae (float): Mean Absolute Error.
        cal_error (float): Calibration error computed with regressor_calibration_error.
        conf (np.array): Confidence levels for the calibration curve.
        acc (np.array): Empirical coverage (accuracy) corresponding to the confidence levels.
    """
    
    # Extract all test data from the loader.
    all_features = []
    all_labels = []
    for batch_x, batch_y in test_loader:
        all_features.append(batch_x)
        all_labels.append(batch_y)
    test_features = torch.cat(all_features, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    
    # Apply the specified shift to the features.
    X_np = test_features.cpu().numpy()
    # print(X_np)
    shifted_X_np = apply_shift(X_np, shift_type, severity)
    # print(shifted_X_np)
    shifted_test_tensor = torch.FloatTensor(shifted_X_np).to(device)
    
    # Create a new DataLoader with the shifted data.
    shifted_test_dataset = TensorDataset(shifted_test_tensor, test_labels.cpu())
    shifted_test_loader = DataLoader(shifted_test_dataset, batch_size=batch_size)
    
    metrics = evaluate_model(model, shifted_test_loader, cfg)

    mae = metrics['mae']
    cal_error = metrics['calibration_error']

    return mae, cal_error


def run_shift_analysis(model, test_loader, cfg, device):
    shift_types = ["gaussian", "mask", "scaling"]
    severity_levels = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for shift in shift_types:
        results[shift] = []
        for sev in severity_levels:
            mae, cal_err = evaluate_under_shift(
                model, test_loader, cfg, cfg.training.batch_size, device, shift, sev
            )
            coverage_val = 1 - (mae / 100)  
            results[shift].append({
                "severity": sev, 
                "mae": mae, 
                "cal_error": cal_err,
                "coverage": coverage_val,
            })
            print(f"Shift: {shift}, Severity: {sev}, MAE: {mae:.4f}, Cal Error: {cal_err:.4f}")
    
    plot_aggregated_boxplot(results, severity_levels, metric="cal_error", ylabel="Calibration Error")
    plot_aggregated_boxplot_accuracy(results, severity_levels, metric="coverage", ylabel="Empirical Coverage")

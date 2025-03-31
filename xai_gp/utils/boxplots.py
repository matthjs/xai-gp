import os
import matplotlib.pyplot as plt


def plot_aggregated_boxplot(results, severity_levels, metric="cal_error", ylabel="Calibration Error"):
    """
    Create a box plot summarizing a given metric across different shift types for each severity level.
    The x-axis treats severity levels as discrete categories (evenly spaced).
    
    Args:
        results (dict): A dictionary with keys for each shift type and values as lists of dicts
                        containing 'severity' and the metric value.
        severity_levels (list): List of severity levels.
        metric (str): The key for the metric to be aggregated from results.
        ylabel (str): The label for the y-axis.
    """
    # Aggregate metric values for each severity level across all shift types.
    aggregated = {sev: [] for sev in severity_levels}
    for shift in results:
        for entry in results[shift]:
            sev = entry["severity"]
            if sev in aggregated:
                aggregated[sev].append(entry[metric])
    
    # Prepare the data for the boxplot.
    data = [aggregated[sev] for sev in severity_levels]
    # Use discrete positions (0, 1, 2, ...) so the boxes are evenly spaced.
    positions = list(range(len(severity_levels)))
    
    # Increase figure size to reduce whitespace.
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                    showfliers=True, whis=[0, 100])
    
    # Color the boxes.
    for box in bp['boxes']:
        box.set(facecolor='lightblue')
    
    # Set x-axis ticks to the discrete positions with labels as the actual severity values.
    ax.set_xticks(positions)
    ax.set_xticklabels(severity_levels)
    ax.set_xlim(-0.5, len(severity_levels)-0.5)
    
    ax.set_xlabel("Shift Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Shift Severity (Aggregated Across Shift Types)")
    
    save_path = os.path.join('../results', 'shift_analysis', f'aggregated_boxplot_{metric}.png')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print(f"Saving aggregated box plot to: {save_path}")
    plt.savefig(save_path)
    plt.show()


def plot_aggregated_boxplot_accuracy(results, severity_levels, metric="coverage", ylabel="Empirical Coverage"):
    """
    Create a box plot for Accuracy across different shift types and severity levels.
    """
    aggregated = {sev: [] for sev in severity_levels}
    for shift in results:
        for entry in results[shift]:
            sev = entry["severity"]
            if sev in aggregated:
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
    ax.set_xlim(-0.5, len(severity_levels)-0.5)
    
    ax.set_xlabel("Shift Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Shift Severity (Aggregated Across Shift Types)")
    
    save_path = os.path.join('../results', 'shift_analysis', 'aggregated_boxplot_coverage.png')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print(f"Saving aggregated coverage box plot to: {save_path}")
    plt.savefig(save_path)
    plt.show()

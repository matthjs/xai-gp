import numpy as np

def apply_shift(X, shift_type, severity):
    """
    Apply a synthetic shift to the data X.
    
    Args:
        X (np.array): Input feature matrix of shape (n_samples, n_features).
        shift_type (str): Type of shift to apply. Options: "gaussian", "mask", "scaling".
        severity (float): Severity level of the corruption (e.g., 0.1 for 10%).
    
    Returns:
        np.array: The shifted feature matrix.
    """
    if shift_type == "gaussian":
        # Add Gaussian noise proportional to the per-feature standard deviation.
        noise = np.random.normal(loc=0, scale=severity * np.std(X, axis=0), size=X.shape)
        return X + noise
    elif shift_type == "mask":
        # Randomly zero out a fraction of the features.
        mask = np.random.rand(*X.shape) < severity
        X_shifted = X.copy()
        X_shifted[mask] = 0
        return X_shifted
    elif shift_type == "scaling":
        # Scale all features by a factor (e.g., 1 + severity).
        scaling_factor = 1 + severity
        return X * scaling_factor
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

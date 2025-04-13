import numpy as np

def gaussian_noise(X, severity):
    """
    Add Gaussian noise to features.
    Noise is drawn from N(0, (severity * std)^2) for each feature.
    """
    noise = np.random.normal(loc=0, scale=severity * np.std(X, axis=0), size=X.shape)
    return X + noise

def masking(X, severity):
    """
    Feature masking: randomly zero-out features.
    Each feature entry is set to zero with probability equal to severity.
    """
    mask = np.random.rand(*X.shape) < severity
    X_shifted = X.copy()
    X_shifted[mask] = 0
    return X_shifted

def scaling(X, severity):
    """
    Feature scaling shift: uniformly scale features.
    A scaling factor of (1 + severity) is applied to all features.
    """
    scaling_factor = 1 + severity
    return X * scaling_factor

def permute_features(X, severity):
    """
    Permute feature values within each column.
    Each entry in a column is replaced with another random value from that column 
    with probability equal to severity.
    """
    X_shifted = X.copy()
    n_rows, n_cols = X.shape
    for j in range(n_cols):
        col_original = X[:, j].copy()
        # Create a permutation of row indices
        permuted_indices = np.random.permutation(n_rows)
        permuted_col = col_original.copy()
        # Determine which entries to replace
        mask = np.random.rand(n_rows) < severity
        permuted_col[mask] = col_original[permuted_indices][mask]
        X_shifted[:, j] = permuted_col
    return X_shifted

def outlier(X, severity):
    """
    Add outlier noise to randomly selected feature entries.
    For each feature, with probability equal to severity, 
    an outlier perturbation is added.
    """
    X_shifted = X.copy()
    # Compute per-feature standard deviation (keep dims for broadcasting)
    std = np.std(X, axis=0, keepdims=True)
    # Outlier noise: large shift with random sign
    noise = np.random.choice([-1, 1], size=X.shape) * (3 * std)
    mask = np.random.rand(*X.shape) < severity
    X_shifted[mask] += noise[mask]
    return X_shifted

def apply_shift(X, shift_type, severity):
    """
    Apply a synthetic shift to the input X.
    
    Supported shift types:
      - "gaussian": Adds Gaussian noise.
      - "mask": Randomly zeros out feature values.
      - "scaling": Scales features uniformly.
      - "permute": Shuffles feature values within columns.
      - "outlier": Injects outlier perturbations into features.
      
    All these methods are applicable to both regression and classification tasks.
    """
    if shift_type == "gaussian":
        return gaussian_noise(X, severity)
    elif shift_type == "mask":
        return masking(X, severity)
    elif shift_type == "scaling":
        return scaling(X, severity)
    elif shift_type == "permute":
        return permute_features(X, severity)
    elif shift_type == "outlier":
        return outlier(X, severity)
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

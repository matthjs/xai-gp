import numpy as np
import pandas as pd
from collections import defaultdict

from baycomp import CorrelatedTTest
from torch.utils.data import Subset, DataLoader
from tqdm.contrib import itertools

from xai_gp.models.ensemble import DeepEnsembleRegressor
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.utils.evaluation import evaluate_model
from xai_gp.utils.training import initialize_model, train_model


def get_pairs(strings):
    pairs = []
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            pairs.append((strings[i], strings[j]))
    return pairs


from sklearn.model_selection import KFold


def statistical_comparison(cfg, train_loader, val_loader, test_loader, input_shape, device) -> None:
    """
    Perform statistical comparison between models using a Bayesian t-test
    after training and evaluating using k-fold cross-validation.
    The statistical tests are done for every metric (e.g., mae, mse, rmse, etc.).
    """
    results = {}
    n_folds = 5  # Set the number of folds for cross-validation

    model_list = ["DeepGPModel", "DSPPModel"]
    model_list.append("DeepEnsembleRegressor" if cfg.data.name == "CASP" else "DeepEnsembleClassifier")

    metrics_list = ['mae', 'mse', 'rmse', 'nll', 'calibration_error', 'sharpness']

    # Step 1: Perform k-fold cross-validation for each model
    for model_type in model_list:
        print(f"\nRunning {model_type}...")

        if model_type == 'DeepGPModel':
            cfg.model = cfg.compare_all.deepgp
            cfg.training = cfg.compare_all.deepgp.training
        elif model_type == 'DSPPModel':
            cfg.model = cfg.compare_all.dspp
            cfg.training = cfg.compare_all.dspp.training
        elif model_type == 'DeepEnsembleRegressor':
            cfg.model = cfg.compare_all.ensemble_mlp
            cfg.training = cfg.compare_all.ensemble_mlp.training
        elif model_type == 'DeepEnsembleClassifier':
            cfg.model = cfg.compare_all.ensemble_cla
            cfg.training = cfg.compare_all.ensemble_cla.training
        else:
            raise ValueError(":(")

        model_metrics = {metric: [] for metric in metrics_list}
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_metrics = {metric: [] for metric in metrics_list}

        # Step 1: Perform k-fold cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
            # Split data for the current fold using Subset
            train_subset = Subset(train_loader.dataset, train_idx)
            val_subset = Subset(train_loader.dataset, val_idx)

            # Create new DataLoader for each fold
            train_fold_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
            val_fold_loader = DataLoader(val_subset, batch_size=val_loader.batch_size, shuffle=False)

            # Initialize and train model
            model, optimizer = initialize_model(cfg, input_shape, device)
            train_model(model, train_fold_loader, optimizer, cfg)

            # Evaluate the model on the validation fold
            metrics = evaluate_model(model, val_fold_loader, cfg)

            # Collect fold-specific metrics
            for metric in metrics_list:
                fold_metrics[metric].append(metrics[metric])

        for metric in metrics_list:
            model_metrics[metric] = fold_metrics[metric]

        results[model_type] = dict(model_metrics)

    # Step 2: Compare models using the Bayesian t-test for every metric
    pairs = get_pairs(model_list)
    for metric in metrics_list:
        print(f"\nComparing models based on {metric}...")
        for pair in pairs:
            print(f"  Comparing {pair[0]} vs {pair[1]} for {metric}...")

            # Perform Bayesian t-test for each metric
            x = np.array(results[pair[0]][metric])
            y = np.array(results[pair[1]][metric])

            bt = CorrelatedTTest.probs(x, y, rope=0)  # Adjust 'rope' as needed
            print(f"  Bayesian t-test result for {pair[0]} vs {pair[1]} on {metric}: {bt}")

    # Step 3: Compute mean and standard deviation for each model's metrics
    data = {}
    for model_type, metrics in results.items():
        for metric, values in metrics.items():
            mean = np.mean(values)
            stddev = np.std(values)
            data[(model_type, metric)] = {'mean': mean, 'stdev': stddev}

    # Convert to a DataFrame for better visualization
    df = pd.DataFrame(data)

    # Transpose to get metrics as rows and predictors as columns
    df = df.T
    print(df)
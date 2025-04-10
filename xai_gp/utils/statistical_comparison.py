import numpy as np
import pandas as pd
from collections import defaultdict

from baycomp import CorrelatedTTest
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


def statistical_comparison(cfg, train_loader, val_loader, test_loader, input_shape, device) -> None:
    """
    Don't really know what to call this function tbh.
    The model's are not loaded but have to be retrained (TODO)
    """
    results = {}
    n_samples = cfg.compare_all.n_samples

    model_list = ["DeepGPModel", "DSPPModel"]
    model_list.append("DeepEnsembleRegressor" if cfg.data.name == "CASP" else "DeepEnsembleClassifier")

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

        model_metrics = defaultdict(list)
        # Statistically better: retrain model at every run
        # Not done here -->
        for run in range(n_samples):
            print(f"  Run {run + 1}/{n_samples}")
            model, optimizer = initialize_model(cfg, input_shape, device)
            train_model(model, train_loader, optimizer, cfg, val_loader=val_loader)
            metrics = evaluate_model(model, test_loader, cfg)

            for k, v in metrics.items():
                model_metrics[k].append(v)

        results[model_type] = dict(model_metrics)

    pairs = get_pairs(model_list)
    for pair in pairs:
        print(pair)
        bt = CorrelatedTTest.probs(np.array(results[pair[0]]['calibration_error']),
                             np.array(results[pair[1]]['calibration_error']), rope=0)
        print(bt)

    data = {}
    for metric, predictors in results.items():
        for predictor, values in predictors.items():
            mean = np.mean(values)
            stddev = np.std(values)
            data[(metric, predictor)] = {'mean': mean, 'stdev': stddev}

    # Convert to a DataFrame for better visualization
    df = pd.DataFrame(data)

    # Transpose to get metrics as rows and predictors as columns
    df = df.T
    print(df)

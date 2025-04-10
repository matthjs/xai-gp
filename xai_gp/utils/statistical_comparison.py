from collections import defaultdict

from xai_gp.models.ensemble import DeepEnsembleRegressor
from xai_gp.models.gp import DeepGPModel, DSPPModel
from xai_gp.utils.evaluation import evaluate_model
from xai_gp.utils.training import initialize_model, train_model


def statistical_comparison(cfg, train_loader, val_loader, test_loader, input_shape, device) -> None:
    """
    Don't really know what to call this function tbh.
    The model's are not loaded but have to be retrained (TODO)
    """
    results = {}
    n_samples = cfg.compare_all.n_samples

    for model_type in ["DeepGPModel", "DSPPModel", "DeepEnsembleRegressor", "DeepEnsembleClassifier"]:
        print(f"\nRunning {model_type}...")

        cfg.model.type = model_type
        model_metrics = defaultdict(list)
        # Statistically better: retrain model at every run
        # Not done here -->
        model, optimizer = initialize_model(cfg, input_shape, device)
        train_model(model, train_loader, optimizer, cfg, val_loader=val_loader)

        for run in range(n_samples):
            print(f"  Run {run+1}/{n_samples}")
            metrics = evaluate_model(model, test_loader, cfg)

            for k, v in metrics.items():
                model_metrics[k].append(v)

        results[model_type] = dict(model_metrics)

    print(results)


# XAI-GP: Explainable AI with Gaussian Processes

A simple implementation of probabilistic machine learning models with uncertainty quantification, focusing on Gaussian Processes and Deep Ensembles.

## Usage

This project uses [Hydra](https://hydra.cc/) for configuration management. To run the code (from the `xai-gp` directory):

```bash
python main.py model=model_config.yaml data=data_config.yaml
```

### Available Models

- `DeepGPModel`: Deep Gaussian Process
- `DSPPModel`: Deep Sigma Point Processes
- `DeepEnsembleRegressor`: Deep Ensemble for regression
- `DeepEnsembleClassifier`: Deep Ensemble for classification

### Available Datasets

- `CASP`: Protein regression dataset
- `CIFAR100`: Image classification dataset (TBD)

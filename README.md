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
- `ESR`: Seizure classification dataset

We provide optimized hyperparameters for the models inside the `conf/models` directory.

## Getting Started
### Prerequisites
*  [Poetry](https://python-poetry.org/).

## Running

You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

## Reproducibility

We save the weights of our optimized model inside the `results/weights` directory.

# License
This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.
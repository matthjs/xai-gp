from typing import List, Dict, Optional, Tuple
import torch.nn.functional as F
import torch
from torch import nn


class TwoHeadMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers_config: Optional[List[Dict[str, int]]] = None):
        """
        Multi-layer perceptron (MLP) with two output heads, configurable through hidden_layers_config.

        :param input_dim: Dimensionality of the input.
        :param output_dim: Dimensionality of each output head.
        :param hidden_layers_config: List defining the configuration of hidden layers.
                                      Each item is a dictionary with:
                                      - "output_dims": Number of units in this layer.
                                      - "activation": Activation function for this layer (e.g., "relu").
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Dynamically create hidden layers based on the provided configuration
        if hidden_layers_config is None:
            self.hidden = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU()
            )
        else:
            for layer_config in hidden_layers_config:
                output_dims = layer_config['output_dims']
                activation = layer_config.get('activation', 'relu')  # Default to ReLU if not specified

                layers.append(nn.Linear(prev_dim, output_dims))

                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")

                prev_dim = output_dims

        # Create the sequential container for the hidden layers
        self.hidden = nn.Sequential(*layers)

        # Two separate output heads
        self.mean_head = nn.Linear(prev_dim, output_dim)  # mean vector
        self.var_head = nn.Linear(prev_dim, output_dim)  # log(stdev) diagonal entries
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Output mean and var prediction.
        :param x: input tensor.
        """
        features = self.hidden(x)
        mean = self.mean_head(features)
        var = F.softplus(self.var_head(features))
        # torch.diag_embed(torch.exp(var)) for covariance matrix
        return mean, var

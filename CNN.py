import torch
import torch.nn as nn
from typing import List, Callable

class CNN(nn.Module):

    DEFAULT_HIDDEN_ACTIVATION_FUNC = torch.relu
    DEFAULT_OUTPUT_ACTIVATION_FUNC = torch.sigmoid

    def __init__(self,
                 layers: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 output_size: int,
                 hidden_activations: List[Callable]=None,
                 output_activation: Callable=None):

        super(CNN, self).__init__()

        # Init layers
        self.hidden = nn.ModuleList()
        for input_size, hidden_output_size, kernel_size, stride in zip(layers, layers[1:], kernel_sizes, strides):
            self.hidden.append(nn.Conv2d(in_channels=input_size, out_channels=hidden_output_size, kernel_size=kernel_size, stride=stride))
        self.fully_connected_layer = None # Placeholder for the fully connected layer, initialized dynamically
        # Initialize hidden activation functions with a default if not provided
        if hidden_activations is None:
            hidden_activations = [self.DEFAULT_HIDDEN_ACTIVATION_FUNC] * (len(layers) - 1)
        elif len(hidden_activations) != len(layers) - 1:
            raise ValueError(f"Number of activation functions must be equal to {len(layers) - 1}")
        self.activations = hidden_activations
        # Initialize output activation function
        if output_activation is None:
            output_activation = self.DEFAULT_OUTPUT_ACTIVATION_FUNC
        self.output_activation = output_activation
        self.output_size = output_size

    def forward(self, x):
        for convolution, activation in zip(self.hidden, self.activations):
            x = activation(convolution(x))

        if self.fully_connected_layer is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fully_connected_layer = nn.Linear(flattened_size, self.output_size)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)

        if getattr(self.output_activation, "__name__", None) == "softmax":
            return self.output_activation(x, dim=1)
        else:
            return self.output_activation(x)
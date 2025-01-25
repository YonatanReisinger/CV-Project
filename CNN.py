import torch
import torch.nn as nn
from typing import List, Callable

class CNN(nn.Module):

    DEFAULT_HIDDEN_ACTIVATION_FUNC = torch.relu

    def __init__(self,
                 layers: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 output_size: int,
                 hidden_activations: List[Callable]=None,
                 output_activation: Callable=None):

        super(CNN, self).__init__()

        # Check if the number of layers is consistent with the number of kernel sizes and strides
        if len(layers) - 1 != len(kernel_sizes) or len(layers) - 1 != len(strides):
            raise ValueError("The number of layers - 1 must be equal to the number of kernel sizes and strides")
        # Check that the output_activation is a function
        if output_activation is not None and not callable(output_activation):
            raise ValueError("Output activation must be a function")

        # Init layers
        self.hidden = nn.ModuleList()
        for input_size, hidden_output_size, kernel_size, stride in zip(layers, layers[1:], kernel_sizes, strides):
            self.hidden.append(nn.Conv2d(in_channels=input_size, out_channels=hidden_output_size, kernel_size=kernel_size, stride=stride))
        self.fully_connected_layer_1 = None # Placeholder for the fully connected layer, initialized dynamically


        # Initialize hidden activation functions with a default if not provided
        if hidden_activations is None:
            hidden_activations = [self.DEFAULT_HIDDEN_ACTIVATION_FUNC] * (len(layers) - 1)
        elif len(hidden_activations) != len(layers) - 1:
            raise ValueError(f"Number of activation functions must be equal to {len(layers) - 1}")
        self.activations = hidden_activations

        self.output_activation = output_activation
        self.output_size = output_size

    def forward(self, x):
        for convolution, activation in zip(self.hidden, self.activations):
            x = activation(convolution(x))

        if self.fully_connected_layer_1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fully_connected_layer_1 = nn.Linear(flattened_size, self.output_size)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)

        if self.output_activation is None:
            return x
        elif getattr(self.output_activation, "__name__", None) == "softmax":
            return self.output_activation(x, dim=1)
        elif getattr(self.output_activation, "__name__", None) == "sigmoid":
            return self.output_activation(x)
        else:
            raise ValueError("Output function should be softmax, sigmoid or without any output function")
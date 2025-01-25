import torch
import torch.nn as nn
from typing import List, Callable

class CNN(nn.Module):

    DEFAULT_HIDDEN_ACTIVATION_FUNC = torch.relu

    def __init__(self,
                 convolution_layers: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 paddings: List[int],
                 output_size: int,
                 hidden_activations: List[Callable]=None,
                 output_activation: Callable=None):

        super(CNN, self).__init__()

        # Initialize hidden activation functions with a default if not provided
        if hidden_activations is None:
            hidden_activations = [self.DEFAULT_HIDDEN_ACTIVATION_FUNC] * (len(convolution_layers) - 1)
        if len(hidden_activations) != len(convolution_layers) - 1:
            raise ValueError(f"Number of activation functions must be equal to {len(convolution_layers) - 1}")
        # Check if the number of layers is consistent with the number of kernel sizes and strides
        if len(convolution_layers) - 1 != len(kernel_sizes) or len(convolution_layers) - 1 != len(strides) or len(convolution_layers) - 1 != len(paddings):
            raise ValueError("The number of convolution layers - 1 must be equal to the number of kernel sizes, strides and paddings")
        # Check that the output_activation is a function
        if output_activation is not None and not callable(output_activation):
            raise ValueError("Output activation must be a function")

        self.activations = hidden_activations
        self.output_activation = output_activation
        self.output_size = output_size

        # Init convolution layers
        self.hidden = nn.ModuleList()
        self.convolution_bns = []
        for input_size, hidden_output_size, kernel_size, stride, padding in zip(convolution_layers, convolution_layers[1:], kernel_sizes, strides, paddings):
            self.hidden.append(nn.Conv2d(in_channels=input_size,
                                         out_channels=hidden_output_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
            self.convolution_bns.append(nn.BatchNorm2d(hidden_output_size))

        # Init fully connected layers
        # Hidden layer 1
        self.fully_connected_layer_1 = None # Placeholder for the fully connected layer, initialized dynamically
        self.fully_connected_layer_1_bn = nn.BatchNorm1d(1000)
        # Hidden layer 2
        self.fully_connected_layer_2 = nn.Linear(1000, 1000)
        self.fully_connected_layer_2_bn = nn.BatchNorm1d(1000)
        # Hidden layer 3
        self.fully_connected_layer_3 = nn.Linear(1000, 1000)
        self.fully_connected_layer_3_bn = nn.BatchNorm1d(1000)
        # Hidden layer 4
        self.fully_connected_layer_4 = nn.Linear(1000, 1000)
        self.fully_connected_layer_4_bn = nn.BatchNorm1d(1000)
        # Final layer
        self.fully_connected_layer_5 = nn.Linear(1000, self.output_size)
        self.fully_connected_layer_5_bn = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        for convolution, convolution_bn, activation in zip(self.hidden, self.convolution_bns, self.activations):
            x = activation(convolution_bn(convolution(x)))

        if self.fully_connected_layer_1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fully_connected_layer_1 = nn.Linear(flattened_size, 1000)

        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer_1(x)
        x = self.fully_connected_layer_1_bn(x)

        x = self.DEFAULT_HIDDEN_ACTIVATION_FUNC(x)
        x = self.fully_connected_layer_2(x)
        x = self.fully_connected_layer_2_bn(x)

        x = self.DEFAULT_HIDDEN_ACTIVATION_FUNC(x)
        x = self.fully_connected_layer_3(x)
        x = self.fully_connected_layer_3_bn(x)

        x = self.DEFAULT_HIDDEN_ACTIVATION_FUNC(x)
        x = self.fully_connected_layer_4(x)
        x = self.fully_connected_layer_4_bn(x)

        x = self.DEFAULT_HIDDEN_ACTIVATION_FUNC(x)
        x = self.fully_connected_layer_5(x)
        x = self.fully_connected_layer_5_bn(x)

        if self.output_activation is None:
            return x
        elif getattr(self.output_activation, "__name__", None) == "softmax":
            return self.output_activation(x, dim=1)
        elif getattr(self.output_activation, "__name__", None) == "sigmoid":
            return self.output_activation(x)
        else:
            raise ValueError("Output function should be softmax, sigmoid or without any output function")
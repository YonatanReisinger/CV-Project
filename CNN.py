import torch
import torch.nn as nn
from typing import List, Callable

class CNN(nn.Module):

    DEFAULT_HIDDEN_ACTIVATION_FUNC = torch.relu

    def __init__(self,
                 hidden_convolution_layers: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 paddings: List[int],
                 hidden_fully_connected_layers: List[int | None],
                 output_size: int,
                 hidden_activations: List[Callable]=None,
                 hidden_fully_connected_activations: List[Callable] = None,
                 output_activation: Callable=None):

        super(CNN, self).__init__()

        # Initialize hidden activation functions with a default if not provided
        if hidden_activations is None:
            hidden_activations = [self.DEFAULT_HIDDEN_ACTIVATION_FUNC] * (len(hidden_convolution_layers) - 1)
        if hidden_fully_connected_activations is None:
            hidden_fully_connected_activations = [self.DEFAULT_HIDDEN_ACTIVATION_FUNC] * (len(hidden_fully_connected_layers) - 1)
        if len(hidden_activations) != len(hidden_convolution_layers) - 1 or len(hidden_fully_connected_activations) != len(hidden_fully_connected_layers) - 1:
            raise ValueError(f"Number of activation functions must be equal to {len(hidden_convolution_layers) - 1}")
        # Check if the number of layers is consistent with the number of kernel sizes and strides
        if len(hidden_convolution_layers) - 1 != len(kernel_sizes) or len(hidden_convolution_layers) - 1 != len(strides) or len(hidden_convolution_layers) - 1 != len(paddings):
            raise ValueError("The number of convolution layers - 1 must be equal to the number of kernel sizes, strides and paddings")
        # Check that the output_activation is a function
        if output_activation is not None and not callable(output_activation):
            raise ValueError("Output activation must be a function")

        self.activations = hidden_activations
        self.output_activation = output_activation
        self.output_size = output_size

        # Init convolution layers
        self.hidden = nn.ModuleList()
        self.convolution_bns = nn.ModuleList()
        for input_size, hidden_output_size, kernel_size, stride, padding in zip(hidden_convolution_layers, hidden_convolution_layers[1:], kernel_sizes, strides, paddings):
            self.hidden.append(nn.Conv2d(in_channels=input_size,
                                         out_channels=hidden_output_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding))
            self.convolution_bns.append(nn.BatchNorm2d(hidden_output_size))

        # Init fully connected layers
        self.fully_connected_layers = nn.ModuleList()
        self.fully_connected_bns = nn.ModuleList()
        # Init the hidden fully connected layers
        for input_size, hidden_output_size in zip(hidden_fully_connected_layers, hidden_fully_connected_layers[1:]):
            if hidden_output_size is None:
                raise ValueError("Output size can not be None")
            if input_size is None:
                self.fully_connected_layers.append(None)
            else:
                self.fully_connected_layers.append(nn.Linear(input_size, hidden_output_size))
            self.fully_connected_bns.append(nn.BatchNorm1d(hidden_output_size))
        # Init output fully connected layer
        if hidden_fully_connected_layers:
            self.fully_connected_layers.append(nn.Linear(hidden_fully_connected_layers[-1], self.output_size))
        else:
            self.fully_connected_layers.append(None)
        self.fully_connected_bns.append(nn.BatchNorm1d(self.output_size))

    def forward(self, x):
        for convolution, convolution_bn, activation in zip(self.hidden, self.convolution_bns, self.activations):
            x = activation(convolution_bn(convolution(x)))

        if self.fully_connected_layers[0] is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            if len(self.fully_connected_layers) >= 2:
                self.fully_connected_layers[0] = nn.Linear(flattened_size, self.fully_connected_layers[1].in_features)
            else:
                self.fully_connected_layers[0] = nn.Linear(flattened_size, self.output_size)

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
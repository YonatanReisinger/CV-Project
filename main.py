import torch
from torchvision import datasets, transforms
from CNN import CNN
from Experiment import Experiment
from typing import List, Callable
import random
from itertools import product
import os

def experiment_architectures(layers_options: List[List[int]],
                             output_activation_options: List[Callable],
                             kernel_sizes_options: List[List[int]],
                             strides_options: List[List[int]]):
    # Generate all possible combinations of parameters
    all_combinations = list(product(
        output_activation_options,
        layers_options,
        kernel_sizes_options,
        strides_options
    ))

    num_combinations = len(all_combinations)
    sampled_combinations = random.sample(all_combinations, k=num_combinations)

    for output_activation, layers, kernel_sizes, strides in sampled_combinations:
        print(f"Testing configuration: Layers={layers}, Kernel Sizes={kernel_sizes}, Strides={strides}, Activation={output_activation.__name__}")
        model = CNN(layers=layers,
                    output_activation=output_activation,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    output_size=10)
        exp = Experiment(model=model,
                         criterion=torch.nn.CrossEntropyLoss(),
                         batch_size=64,
                         epochs=40,
                         lr=0.1,
                         optimizer_name="SGD")
        exp()
        exp.to_pickle()
        print(f"\n--------{exp.score}--------")

def experiment_shallow_architectures():
    layers_options = [
        [3, 32, 64],  # Small model: Input (RGB), 32 filters, then 64 filters
        [3, 64, 128],  # Medium model
    ]
    output_activation_options = [torch.nn.functional.softmax]
    kernel_sizes_options = [
        [3, 3],  # Standard 3x3 kernels
        [5, 5],  # Larger receptive fields
    ]
    strides_options = [
        [1, 2],  # First layer standard, second layer downsampling
    ]
    experiment_architectures(layers_options, output_activation_options, kernel_sizes_options, strides_options)

def load_experiments(directory="."):
    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl") and f.startswith("Experiment")]
    loaded_data = {}

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        try:
            loaded_data[pkl_file] = Experiment.from_pickle(file_path)
        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")

    return loaded_data

def main():
    # experiment_shallow_architectures()
    loaded_data = load_experiments()
    for name, data in loaded_data.items():
        print(data)

if __name__ == '__main__':
    main()
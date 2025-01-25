import torch
from torchvision import datasets, transforms
from CNN import CNN
from Experiment import Experiment
from typing import List, Callable
import random
from itertools import product
import os

def experiment_architectures(layers_options: List[List[int]],
                             output_activation_options: List[Callable | None],
                             kernel_sizes_options: List[List[int]],
                             strides_options: List[List[int]],
                             epochs: int,
                             optimizer_name: str,
                             lr: float,
                             momentum: float = 0,
                             probability: float = 0.5):

    # Generate all possible combinations of parameters
    all_combinations = list(product(
        output_activation_options,
        layers_options,
        kernel_sizes_options,
        strides_options
    ))

    if probability > 1 or probability < 0:
        raise ValueError("probability needs to be between 0 and 1")
    num_combinations = len(all_combinations)
    sampled_combinations = random.sample(all_combinations, k=int(num_combinations * probability))

    for output_activation, layers, kernel_sizes, strides in sampled_combinations:
        print(f"Testing configuration: Layers={layers}, Kernel Sizes={kernel_sizes}, Strides={strides}, Activation={output_activation.__name__ if output_activation else None}, Optimizer={optimizer_name}")
        model = CNN(convolution_layers=layers,
                    output_activation=output_activation,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    output_size=10)
        exp = Experiment(model=model,
                         criterion=torch.nn.CrossEntropyLoss(),
                         batch_size=100,
                         epochs=epochs,
                         lr=lr,
                         momentum=momentum,
                         optimizer_name=optimizer_name)
        exp()
        exp.to_pickle()
        print(f"\n--------{exp.score}--------")

def experiment_shallow_architectures():
    layers_options = [
        [3, 32, 64],
        [3, 64, 128],
    ]
    output_activation_options = [torch.nn.functional.softmax]
    kernel_sizes_options = [
        [3, 3],
        [5, 5],
    ]
    strides_options = [
        [1, 2],
    ]
    experiment_architectures(layers_options, output_activation_options, kernel_sizes_options, strides_options, 20, "SGD")

def experiment_architectures_with_depth_4():
    layers_options = [
        [3, 32, 64, 128],
    ]
    output_activation_options = [None]
    kernel_sizes_options = [
        [3, 5, 5],
    ]
    strides_options = [
        [1, 2, 2],
    ]
    experiment_architectures(layers_options=layers_options,
                             output_activation_options=output_activation_options,
                             kernel_sizes_options=kernel_sizes_options,
                             strides_options=strides_options,
                             epochs=50,
                             optimizer_name="SGD",
                             momentum=0.2,
                             probability = 1)

def experiment_architectures_with_depth_5():
    layers_options = [
        [3, 32, 64, 128, 256],
        [3, 64, 128, 256, 512],
    ]
    output_activation_options = [torch.nn.functional.softmax]
    kernel_sizes_options = [
        [5, 5, 5, 5],
    ]
    strides_options = [
        [1, 1, 1, 1],
    ]
    experiment_architectures(layers_options, output_activation_options, kernel_sizes_options, strides_options, 60, "SGD", 1)

def experiment_architectures_with_depth_4_new():
    layers_options = [
        [3, 32, 64, 128],
    ]
    output_activation_options = [None]
    kernel_sizes_options = [
        [5, 5, 5],
    ]
    strides_options = [
        [1, 2, 2],
    ]
    experiment_architectures(layers_options=layers_options,
                             output_activation_options=output_activation_options,
                             kernel_sizes_options=kernel_sizes_options,
                             strides_options=strides_options,
                             epochs=40,
                             optimizer_name="SGD",
                             lr=0.1,
                             momentum=0.2,
                             probability = 1)

def load_experiments(directory="."):
    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl") and f.startswith("Experiment_4_") and "2025-01-25" in f]
    loaded_data = {}

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        try:
            loaded_data[pkl_file] = Experiment.from_pickle(file_path)
        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")

    return loaded_data

def load_experiments_and_print():
    loaded_data = load_experiments()
    for name, data in loaded_data.items():
        print(data)

def main():
    # experiment_architectures_with_depth_4_new()
    load_experiments_and_print()


if __name__ == '__main__':
    main()
    # experiment_shallow_architectures()
    # exp = Experiment.from_pickle("Experiment_25.68_2025-01-24_23:53.pkl")
    # print(exp)
import torch
from torchvision import datasets, transforms
from CNN import CNN
from CNN_V2 import CNN_V2
from Experiment import Experiment
from typing import List, Callable
import random
from itertools import product
import os

def experiment_architectures(layers_options: List[List[int]],
                             output_activation_options: List[Callable | None],
                             kernel_sizes_options: List[List[int]],
                             strides_options: List[List[int]],
                             paddings_options: List[List[int]],
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
        strides_options,
        paddings_options
    ))

    if probability > 1 or probability < 0:
        raise ValueError("probability needs to be between 0 and 1")
    num_combinations = len(all_combinations)
    sampled_combinations = random.sample(all_combinations, k=int(num_combinations * probability))

    for output_activation, layers, kernel_sizes, strides, paddings in sampled_combinations:
        print(f"Testing configuration: Layers={layers}, Kernel Sizes={kernel_sizes}, Strides={strides}, Activation={output_activation.__name__ if output_activation else None}, Optimizer={optimizer_name}")
        model = CNN(convolution_layers=layers,
                    output_activation=output_activation,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    paddings=paddings,
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
    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl") and f.startswith("Experiment_5_")]
    loaded_data = {}

    for pkl_file in pkl_files:
        file_path = os.path.join(directory, pkl_file)
        try:
            loaded_data[pkl_file] = Experiment.from_pickle(file_path)
            print(loaded_data[pkl_file])
        except Exception as e:
            print(f"Failed to load {pkl_file}: {e}")

    return loaded_data

def load_experiments_and_print():
    loaded_data = load_experiments()
    for name, data in loaded_data.items():
        print(data)

def experiment5():
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 2, 2],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=100,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment5_2():
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 2, 2],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.2)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment5_3():
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment5_4():
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 2, 2],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.01,
                p_fc=0.2)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment5_5():
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 2, 2],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.2)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=50,
                     lr=0.2,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_1():
    # bigger pooling, more epochs
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[3, 3, 3],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=300,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_2():
    # Higher learning rate
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.2,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_3():
    # Much Higher momentum
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.1,
                     momentum=0.9,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_4():
    # Higher momentum
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.1,
                     momentum=0.4,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_5():
    # bigger batch size
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=256,
                     epochs=200,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_6():
    model = CNN_V2(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=200,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def experiment6_7():
    # smaller batch size
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 1, 1],
                paddings=[2, 2, 2],
                max_pool_kernel_sizes=[2, 2, 2],
                output_size=10,
                p_conv=0.2,
                p_fc=0.5)
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=64,
                     epochs=200,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def main():
    experiment6_1()
    experiment6_2()
    experiment6_3()
    experiment6_4()
    experiment6_5()
    experiment6_6()
    experiment6_7()


if __name__ == '__main__':
    # main()
    # experiment_shallow_architectures()
    # exp = Experiment.from_pickle("Experiment_7_81.88_2025-01-27_16:28_SGD.pkl")
    # exp.plot_loss()
    # print(exp)
    load_experiments()
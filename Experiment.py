import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, List, Tuple, Dict
import torch.optim as optim
import copy
import torch
import pickle
import datetime
import pandas as pd

def train(optimizer, epochs, model, train_loader, criterion, lr, to_print = True) -> Tuple[List[float], List[Dict[str, torch.Tensor]]]:
    LOSS = []
    models_states = []
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        models_states.append(copy.deepcopy(model.state_dict()))
        if to_print:
          print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
    return LOSS, models_states

def accuracy(net, test_loader) -> float:
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = net(data)
            outputs = outputs.squeeze()
            predicted = outputs.max(dim=1).indices
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

class Experiment:
    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 batch_size: int,
                 epochs: int,
                 lr: float,
                 optimizer_name: str,
                 train: DataLoader=None,
                 test: DataLoader=None):

        self.model = model
        if train is None and test is None:
            train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
            test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        else:
            self.train_loader = train
            self.test_loader = test

        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.LOSS = None
        self.score = None
        self.models_states = None
        if optimizer_name == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer has to be SGD or Adam")

    def __call__(self) -> None:
        self.LOSS, self.models_states = train(self.optimizer, self.epochs, self.model, self.train_loader, self.criterion, self.lr)
        self.score = accuracy(self.model, self.test_loader)

    def __repr__(self) -> str:
        layers = [layer.out_channels for layer in self.model.hidden]
        layers.insert(0, self.model.hidden[0].in_channels)
        # retrieve kernel sizes and stride
        kernel_sizes = [layer.kernel_size[0] for layer in self.model.hidden]
        strides = [layer.stride[0] for layer in self.model.hidden]
        # Prepare data for DataFrame
        data = {
            "Convolution Layers": str(layers),
            "Kernels Sizes": str(kernel_sizes),
            "Strides": str(strides),
            "Convolution Activations": str([act.__name__ for act in self.model.activations]),
            "Output Function": self.model.output_activation.__name__,
            "Output Size": self.model.output_size,
            # "Optimizer": self.optimizer.__class__.__name__,
            "Criterion": self.criterion.__class__.__name__,
            "Epochs": self.epochs,
            "Learning Rate": self.lr,
            "Batch Size": self.batch_size,
            "Loss": f'{self.LOSS[-1]:.2f}'
        }

        # Convert to DataFrame, setting the parameter descriptions as the index
        df = pd.DataFrame(list(data.items()), columns=['Parameter', 'Value']).set_index('Parameter')

        # Construct the final representation
        result = "\n---------- Experiment Results ----------\n"
        result += "   --- Hyper Parameters ---\n"
        result += str(df) + "\n"
        result += "   --- Final Score ---\n"
        result += f"{f'{self.score:.2f}' if self.score is not None else 'Not Evaluated'}"
        return result

    def to_pickle(self, file_path: str = None) -> None:
        if file_path is None:
            # Generate file name with the class name, current date-time, and the score
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
            score_str = f"_{self.score:.2f}" if hasattr(self, 'score') and self.score is not None else ""
            file_path = f"{self.__class__.__name__}{score_str}_{current_time}.pkl"

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_pickle(file_path: str) -> 'Experiment':
        with open(file_path, 'rb') as file:
            loaded_experiment = pickle.load(file)
        return loaded_experiment




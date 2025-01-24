import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, List, Tuple, Dict
import torch.optim as optim
import copy
import torch
import pickle
import datetime

def train(epochs, model, train_loader, criterion, lr, to_print = False) -> Tuple[List[float], List[Dict[str, torch.Tensor]]]:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    LOSS = []
    models_states = []
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1))
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
            predicted = (outputs > 0.5).float()
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

    def __call__(self) -> None:
        self.LOSS, self.models_states = train(self.epochs, self.model, self.train_loader, self.criterion, self.lr)
        self.score = accuracy(self.model, self.test_loader)

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



